// proto/validation — ClusterCache 단위검증. CHECKLIST §3 cache 4 체크.
//  ① 잔여≥크기 admit / 넘으면 거부(SSD行)  ② evict_age 초과 HBM→SSD 강등
//  ③ 적격임계 미만 비-HBM(SSD 만)          ④ hit/SSD/recompute 비용 분기 + 카운터.
#include "scheduler/cache.h"

#include <cstdio>

static int g_fail = 0;
#define CHECK(cond, msg)                                          \
    do {                                                          \
        if (!(cond)) {                                            \
            std::printf("FAIL: %s (line %d)\n", (msg), __LINE__); \
            g_fail = 1;                                           \
        }                                                         \
    } while (0)

using namespace proto;

// 작은 구체 수치: kv_bpt=1 (바이트 = 길이), 노드 용량 = 30 (적격 엔트리 ~2개분),
// 적격임계 = 5, evict_age = 10, gone_age = 100, offload_bw = 4.
static CacheConfig make_cfg() {
    return CacheConfig{/*eligibility_threshold=*/5, /*evict_age=*/10,
                       /*gone_age=*/100, /*offload_bw_bytes_per_round=*/4.0};
}

// ① admit when fits / reject when over capacity.
static void test_admit_capacity() {
    ClusterCache c(/*num_nodes=*/1, /*node_capacity_bytes=*/30, /*kv_bpt=*/1, make_cfg());
    // 적격 엔트리(len=20, bytes=20) 채움 → HBM, used=20.
    c.on_complete(0, /*id=*/1, /*len=*/20, /*now=*/0);
    CHECK(c.node_used(0) == 20, "fitting entry occupies HBM bytes");
    // 다음 적격(len=20, bytes=20) → 20+20=40 > 30 → 거부 → SSD, used 불변.
    c.on_complete(0, /*id=*/2, /*len=*/20, /*now=*/0);
    CHECK(c.node_used(0) == 20, "over-capacity entry does not increase used");
    CHECK(c.node_used(0) <= c.node_capacity(), "used never exceeds capacity");
    // 들어간 것 = Hit, 넘친 것 = MissReload(SSD).
    auto in = c.lookup(1, 1);
    CHECK(in.result == CacheResult::Hit, "fitting id hits");
    auto over = c.lookup(2, 1);
    CHECK(over.result == CacheResult::MissReload, "over-capacity id is SSD (MissReload)");
}

// ② evict_age → HBM 강등 SSD.
static void test_evict_age() {
    ClusterCache c(1, 30, 1, make_cfg());
    c.on_complete(0, 1, 20, /*now=*/0);
    // 즉시 조회 = Hit.
    auto h = c.lookup(1, /*now=*/0);
    CHECK(h.result == CacheResult::Hit, "immediate lookup hits");
    CHECK(c.node_used(0) == 20, "HBM bytes present before evict");
    // now 를 last(=0) + evict_age(10) 초과로 진행 후 tick.
    c.evict_tick(/*now=*/11);  // 11 - 0 > 10 → 강등.
    CHECK(c.node_used(0) == 0, "node_used dropped after evict demote");
    auto r = c.lookup(1, /*now=*/12);
    CHECK(r.result == CacheResult::MissReload, "post-evict lookup is MissReload (SSD)");
}

// ③ ineligible(short) → HBM 미캐싱, SSD 만, used 불변.
static void test_ineligible() {
    ClusterCache c(1, 30, 1, make_cfg());
    // len=5 == threshold → 적격 아님(>threshold 필요).
    c.on_complete(0, 1, /*len=*/5, /*now=*/0);
    CHECK(c.node_used(0) == 0, "ineligible entry does not occupy HBM");
    auto r = c.lookup(1, /*now=*/1);
    CHECK(r.result == CacheResult::MissReload, "ineligible id is SSD, never Hit");
}

// ④ cost 분기 정확 + 카운터.
static void test_cost_branches() {
    ClusterCache c(1, 30, 1, make_cfg());  // offload_bw = 4.
    c.on_complete(0, /*id=*/1, /*len=*/20, /*now=*/0);  // HBM
    c.on_complete(0, /*id=*/2, /*len=*/8, /*now=*/0);   // len=8 적격, 20+8=28<=30 → HBM

    // Hit → reload_cost == 0.
    auto hit = c.lookup(1, /*now=*/1);
    CHECK(hit.result == CacheResult::Hit && hit.reload_cost == 0.0, "Hit reload_cost == 0");
    CHECK(c.hbm_hits() == 1, "hbm_hits incremented");

    // id 2 를 SSD 로 강등(idle 시키고 tick). id 1 은 방금 조회(now=1)라 살아있게,
    // id 2 만 늙히려 별도 캐시로 SSD 비용 검증.
    ClusterCache c2(1, 30, 1, make_cfg());
    c2.on_complete(0, /*id=*/9, /*len=*/8, /*now=*/0);  // HBM, bytes=8
    c2.evict_tick(/*now=*/11);                          // 11-0>10 → SSD
    auto reload = c2.lookup(9, /*now=*/12);
    // SSD reload_cost = len*kv_bpt/offload_bw = 8*1/4 = 2.0 (정확).
    CHECK(reload.result == CacheResult::MissReload, "demoted entry is MissReload");
    CHECK(reload.reload_cost == 2.0, "SSD reload_cost == len*kv_bpt/offload_bw");
    CHECK(c2.ssd_reloads() == 1, "ssd_reloads incremented");

    // gone_age 후 → MissRecompute. last 는 위 lookup 에서 12 로 리셋됨.
    // SSD idle > gone_age(100): now - 12 > 100 → now >= 113.
    c2.evict_tick(/*now=*/113);
    auto gone = c2.lookup(9, /*now=*/114);
    CHECK(gone.result == CacheResult::MissRecompute, "post-gone_age lookup is MissRecompute");
    CHECK(gone.node == -1, "recompute node == -1");
    CHECK(gone.reload_cost == 0.0, "recompute reload_cost == 0");
    CHECK(c2.recomputes() == 1, "recomputes incremented");

    // 미기록 id → MissRecompute, 카운터 증가.
    auto miss = c2.lookup(/*id=*/777, /*now=*/115);
    CHECK(miss.result == CacheResult::MissRecompute, "unknown id is MissRecompute");
    CHECK(c2.recomputes() == 2, "recomputes incremented for unknown id");
}

int main() {
    test_admit_capacity();
    test_evict_age();
    test_ineligible();
    test_cost_branches();
    if (g_fail) {
        std::printf("test_cache FAILED\n");
        return 1;
    }
    std::printf("test_cache OK\n");
    return 0;
}
