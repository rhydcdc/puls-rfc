// PULS-ENGINE — 실 서빙 런타임 드라이버. CONTRACT.md §2 (실 서빙 = QueueSource) / §5.4.
//
// 경계: runtime 은 실 트래픽 지향이다. 분포 B / RNG / 무한풀 emulation 을 쓰지 않는다.
//       요청은 실제 대기 큐(std::vector<PulledRequest>)에서 온다. core 의 스케줄링 로직
//       (NodeScheduler / GlobalScheduler)은 RequestSource 로만 새 요청을 얻으므로,
//       여기서는 그 RequestSource 를 실 큐(QueueSource)로 구현해 끼워 넣는다.
//
// 이 파일은 main() 을 갖지 않는다(CMake 가 main_runtime.cpp 와 함께 빌드 — 다중 main 금지).
// QueueSource / PulsRuntime 정의 + 단일 노출 함수 run_runtime_demo() 만 둔다.
// 헤더 새 파일을 만들지 않는다(드라이버 내부 선언). main_runtime.cpp 는 run_runtime_demo()
// 한 함수만 호출한다.

#include "core/derive.h"
#include "core/node_scheduler.h"
#include "core/operating_point.h"
#include "core/request_source.h"

#include <algorithm>
#include <cstdio>
#include <vector>

namespace puls {

// ─────────────────────────────────────────────────────────────────────────────
// QueueSource — RequestSource 의 실 서빙 구현. 실제 대기 요청 벡터를 보유한다.
//   pull_near(ideal, cap_room): 큐에서 cap_room 적합하며 |length − ideal| 최소인 실 요청을
//   꺼내(제거) 반환. 없으면 PulledRequest{prompt=-1}. 분포 B / RNG 미사용 — 실 트래픽 지향.
// ─────────────────────────────────────────────────────────────────────────────
class QueueSource : public RequestSource {
public:
    // 실 대기 요청을 큐에 넣는다. dtot = 요청의 총 decode 길이(런타임에선 실제 max_tokens).
    void enqueue(int prompt, int dtot) { queue_.push_back(PulledRequest{prompt, dtot}); }

    int waiting() const { return static_cast<int>(queue_.size()); }

    // ideal 근방(cap 적합)에서 실 요청 한 개를 큐에서 꺼낸다. 없으면 prompt<0.
    PulledRequest pull_near(double ideal_tokens, long long cap_room_tokens) override {
        int best = -1;
        double best_dev = 0.0;
        for (int i = 0; i < static_cast<int>(queue_.size()); ++i) {
            if (static_cast<long long>(queue_[i].prompt) > cap_room_tokens) continue;
            double dev = std::abs(static_cast<double>(queue_[i].prompt) - ideal_tokens);
            if (best < 0 || dev < best_dev) {
                best_dev = dev;
                best = i;
            }
        }
        if (best < 0) return PulledRequest{-1, 0};
        PulledRequest r = queue_[best];
        queue_.erase(queue_.begin() + best);  // 꺼냈으니 큐에서 제거(실 소비)
        return r;
    }

private:
    std::vector<PulledRequest> queue_;
};

// ─────────────────────────────────────────────────────────────────────────────
// PulsRuntime — 실 서빙 골격. 설정(ModelSpec/HwSpec/prefill)으로 derive 해 OperatingPoint 를
//   보유하고, 그 동작점으로 QueueSource 를 출처 삼아 NodeScheduler 를 구동한다.
//   (드라이버이므로 얇다 — 핵심 로직은 전부 core.)
// ─────────────────────────────────────────────────────────────────────────────
class PulsRuntime {
public:
    PulsRuntime(const ModelSpec& model, const HwSpec& hw, int prefill_tokens)
        : op_(derive_operating_point(model, hw, prefill_tokens)),
          node_(op_, queue_) {}

    const OperatingPoint& op() const { return op_; }
    QueueSource& queue() { return queue_; }
    NodeScheduler& node() { return node_; }

    // 실 큐에서 센터링 admit 으로 노드 풀을 동작점 decode_pool 근처까지 채운다(cold-start 대용).
    // 큐가 마르거나 풀이 차면 멈춘다 — 분포 B 무한풀 가정 없음(실 트래픽).
    void warm_fill() {
        int guard = 0;
        const int budget = op_.decode_pool + queue_.waiting() + 4;
        while (static_cast<int>(node_.pool().size()) < op_.decode_pool &&
               queue_.waiting() > 0 && guard++ < budget) {
            std::size_t before = node_.pool().size();
            node_.admit_centered(/*fresh=*/false);
            if (node_.pool().size() == before) break;  // 적합 요청 없음 — 중단
        }
    }

private:
    OperatingPoint op_;
    QueueSource    queue_;
    NodeScheduler  node_;
};

// ─────────────────────────────────────────────────────────────────────────────
// 데모: Llama70B + B200 + prefill 128 동작점 인쇄 + 실 큐에 합성 요청을 넣고
//   노드 스케줄 1~2 라운드를 돌려 "실 서빙 경로가 동작함"을 보인다. 분포 B 미사용.
//   (합성 요청은 데모 입력일 뿐 — 분포 B 샘플러나 RNG 가 아니라 고정 리스트다.)
// ─────────────────────────────────────────────────────────────────────────────
void run_runtime_demo() {
    ModelSpec llama{/*layers*/80, /*hidden*/8192, /*heads*/64,
                    /*kv_heads*/8, /*head_dim*/128, /*ffn_inter*/28672};
    HwSpec b200{/*tflops*/2200.0, /*mfu*/0.6};

    PulsRuntime rt(llama, b200, /*prefill_tokens*/128);
    const OperatingPoint& op = rt.op();

    std::printf("===== PULS 실 서빙 런타임 (QueueSource — 실 대기 큐, 분포 B 미사용) =====\n");
    std::printf("동작점 (derive: Llama70B + B200 + prefill 128):\n");
    std::printf("  ctx_balance        = %.0f tokens\n", op.ctx_balance);
    std::printf("  decode_count_target= %d  (N_dec)\n", op.decode_count_target);
    std::printf("  kv_operating_target= %lld\n", op.kv_operating_target);
    std::printf("  ffn_batch          = %d\n", op.ffn_batch);
    std::printf("  balance_time_us X  = %.2f us\n", op.balance_time_us);
    std::printf("  decode_pool        = %d   node_min=%d node_max=%d\n",
                op.decode_pool, op.node_min, op.node_max);
    std::printf("  instance_a_tb      = %.2f TB   hbm_fits=%d\n\n",
                op.instance_a_tb, static_cast<int>(op.hbm_fits));

    // 실 큐에 합성 대기 요청을 적재 — ctx_balance 근방을 노린 고정 길이들(고정 리스트, RNG 0).
    // decode_pool 보다 넉넉히 넣어 센터링/힐링이 큐에서 골라 쓰는 모습을 보인다.
    const int c = static_cast<int>(op.ctx_balance);
    std::vector<int> prompts;
    for (int k = 0; k < op.decode_pool + 20; ++k) {
        // ctx_balance 주변에 퍼진 합성 길이(작은~큰). 실서빙에선 이 자리에 실 프롬프트 길이가 온다.
        int len = c / 2 + (k % 5) * (c / 4);  // 0.5c, 0.75c, 1.0c, 1.25c, 1.5c 순환
        prompts.push_back(len);
    }
    for (std::size_t k = 0; k < prompts.size(); ++k) {
        // dtot: 요청의 총 decode 길이(실서빙 max_tokens 자리). 데모는 작은 고정 budget(1~2)로
        // 두어 2 라운드 안에 일부가 완료 retire → per-completion 힐링이 큐에서 보충하는 모습을 보인다.
        int dtot = 1 + static_cast<int>(k % 2);   // 1 또는 2 토큰 decode budget
        rt.queue().enqueue(prompts[k], dtot);
    }
    std::printf("실 대기 큐에 합성 요청 %d개 적재 (고정 리스트 — 분포 B/RNG 아님).\n",
                rt.queue().waiting());

    // 센터링 admit 으로 큐에서 골라 노드 풀 구성(QueueSource 가 ideal 최근접 실 요청 제공).
    rt.warm_fill();
    std::printf("센터링 admit 후: 노드 풀 %zu개, 큐 잔여 %d개, mean_live_kv=%.0f (타깃 ctx %.0f)\n",
                rt.node().pool().size(), rt.queue().waiting(),
                rt.node().mean_live_kv(), op.ctx_balance);

    // 노드 스케줄 라운드 — 2 μ-batch advance + 완료 retire + per-completion 힐링(큐에서 보충).
    for (int round = 1; round <= 2; ++round) {
        int done = rt.node().advance_round();
        std::printf("라운드 %d: 완료 retire %d개, 풀 %zu개, 큐 잔여 %d개, mean_live_kv=%.0f\n",
                    round, done, rt.node().pool().size(),
                    rt.queue().waiting(), rt.node().mean_live_kv());
    }
    std::printf("→ 실 서빙 경로 동작 확인: derive→센터링 admit→μ-batch steering→힐링이\n"
                "  모두 QueueSource(실 큐)에서 요청을 당겼다. 분포 B 진입 0.\n");
}

}  // namespace puls
