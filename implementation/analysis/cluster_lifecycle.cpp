// 통합 lifecycle sim (검증된 조각의 통합, PULS 독립·composition 명중만).
//
// 한 sim 으로: 콜드스타트(KV 센터) → 운영[프리필 steering+age-cap → (종속성) 완료시 디코드로 전이
//   → 디코드 steering+age-cap → 완료 → per-completion 힐링] → 프리필 풀 보충. age-cap=5.
// 검증 목표: 프리필→디코드 종속성·age-cap 넣어도 디코드·프리필 composition 이 유지되는가.
//
// ★ 결과(배포 동작점 prefill 128, OPERATING_POINT §4.1): 디코드(62∧6.15M) 100%·Σdev 0.20%,
//   프리필(128∧12.8M-work) 100%·Σdev 0.07%. 로직(steering·greedy·healing·age-cap·KV센터링)은
//   스케일 불변 — 동작점은 상수만 바뀐다(env PREFILL 로 256 등 스윕 가능).
//
// 동역학(검증 반영):
//  - 디코드 풀 = live KV(prompt+dec) 평균을 100K 로 직접 센터(admitDecodeCentered). 한 노드 내
//    디코딩 누적으로 footprint 가 prompt×1.24 로 뜨므로, admit 을 prompt 가 아니라 *live KV* 기준
//    피드백(ideal=100K×(cnt+1)−liveSum)으로 당겨야 12.3M 을 123개 전에 안 넘긴다(count-miss 해소).
//  - 프리필 풀 = 독립 abundant·다양 depth. depth-0 admit, depth-work steering+age-cap. 완료 → 디코드 전이.
//    ★ 프리필 풀은 advance 폭(~12/round)에 맞춰 작게(PF_POOL≈60): 크면 노드 내 aging 잉여가 커져
//      age-cap 이 얕은 강제토큰으로 배치를 메워 depth 추락(over-fire). 잉여 과대가 위험, 과소 아님.
//  - 잉여 = 노드 내 (풀 − 2 μ-batch). 디코드 잉여 10 = Σdev plateau knee(대형모델 적재 위해 최소).
//  - throughput 균형: avg_dtot ≈ avg_prompt × N_dec/prefill (= 123/256 = 62/128, 전이율 ≈ 완료율).
//
// 빌드: g++ -O2 -std=c++17 analysis/cluster_lifecycle.cpp -o analysis/cluster_lifecycle.exe
// 실행(배포): ./cluster_lifecycle.exe   |  스윕: PREFILL=256 PF_POOL=150 DEC_POOL=256 AGE_CAP=5 ./...
#include <bits/stdc++.h>
using namespace std;

static const int       Z         = 16;
static const double    TGT_AVG   = 100000.0;
static int             DEC_CNT   = 123;          // 동작점 스케일 (env PREFILL 로 128↔256 등)
static long long       DEC_KV    = 12300000LL;
static int             PF_TOK    = 256;
static long long       PF_WORK   = 25600000LL;
static int             DEC_POOL  = 300;          // 동작점: 2 μ-batch(246) + 잉여 54
static int             PF_POOL   = 300;          // 프리필 풀(abundant — env PF_POOL 로 스윕)
static int             AGE_CAP   = 5;            // 배치 구성 fairness (env AGE_CAP 로 스윕)
static const double    EDGE_BAND = 1000.0;

static mt19937 rng(7);
static inline double U(){ return uniform_real_distribution<double>(0.0,1.0)(rng); }
static inline int logU(double lo,double hi){ return (int)exp(U()*(log(hi)-log(lo))+log(lo)); }
static int sampleB(){ double u=U();
  if(u<0.20) return logU(1000,16000);
  if(u<0.90) return logU(16000,256000);
  return logU(256000,1000000); }
// throughput 균형: avg_dtot ≈ avg_prompt × 123/256
static int sampleDtot(int p){ return max(1000,(int)(p*(123.0/256.0)*(0.6+0.8*U()))); }
// 엣지 게이트: 풀 평균 100K 로(긴 것 shed) — 센터 프롬프트 반환
static int centeredPrompt(){
  static vector<int> buf; static int idx=0;
  if(idx>=(int)buf.size()){
    vector<int> w; for(int i=0;i<20000;i++) w.push_back(sampleB());
    sort(w.rbegin(),w.rend()); long long s=0; for(int x:w)s+=x; int c=w.size(),i=0;
    while(c>0 && (double)s/c > TGT_AVG+EDGE_BAND){ s-=w[i]; c--; i++; }
    buf.assign(w.begin()+i,w.end()); shuffle(buf.begin(),buf.end(),rng); idx=0;
  }
  return buf[idx++];
}

struct Req { int prompt,dtot,pf,dec,wait; };

struct Node { vector<Req> dec, pf, ready; };  // 디코딩 / 프리필링 / 전이대기(완료 프리필)

// 디코드 풀에 1개 admit — live KV(prompt+dec) 평균을 100K 로 직접 센터(프롬프트 X).
// ideal = 100K×(cnt+1) − liveSum = 새 평균이 정확히 100K 되는 footprint. 풀이 떠 있으면 ideal↓ → 작은 것.
// fresh=true(힐링, dec=0) / false(콜드스타트, dec 랜덤 진행). best-of-200 풀 샘플.
static void admitDecodeCentered(Node& nd, bool fresh){
  long long liveSum=0; for(auto&q:nd.dec) liveSum+=(long long)q.prompt+q.dec;
  int cnt=(int)nd.dec.size();
  double ideal=TGT_AVG*(cnt+1)-(double)liveSum; if(ideal<1000) ideal=1000;
  int bp=-1; double bd=1e18;
  for(int k=0;k<200;k++){ int p=sampleB(); double d=fabs((double)p-ideal); if(d<bd){bd=d;bp=p;} }
  Req q; q.prompt=bp; q.dtot=sampleDtot(bp); q.pf=bp; q.dec=fresh?0:(int)(U()*q.dtot); q.wait=0;
  nd.dec.push_back(q);
}

// 디코드 한 μ-batch 구성: 공유 used 에서 123 (12.3M) 최근접 + age-cap. picked 추가, hit/Σ.
// (한 노드는 2 μ-batch 를 돌리므로 이걸 used 공유로 2번 호출 → 246 advance, 잉여만 aging)
static bool composeDecodeBatch(vector<Req>&d, vector<char>&used, vector<int>&picked, double&dev){
  int m=d.size(); picked.clear();
  long long S=0; int n=0;
  while(n<DEC_CNT && S<DEC_KV){
    int sel=-1;
    for(int i=0;i<m;i++) if(!used[i] && d[i].wait>=AGE_CAP){ sel=i; break; }       // age-cap(노드 내)
    if(sel<0){ long long ideal=(long long)llround((double)(DEC_KV-S)/(DEC_CNT-n)); double bd=1e18;
      for(int i=0;i<m;i++) if(!used[i]){ double x=fabs((double)(d[i].prompt+d[i].dec)-ideal); if(x<bd){bd=x;sel=i;} } }
    if(sel<0) break; used[sel]=1; picked.push_back(sel); S+=d[sel].prompt+d[sel].dec; n++;
  }
  dev = fabs((double)S-(double)DEC_KV)/DEC_KV;
  return abs(n-DEC_CNT)<=12 && dev<=0.10;
}

// 프리필 steering: pf 풀서 256토큰 (25.6M depth-work) 최근접 + age-cap(chunk==0 spread).
static bool composePrefill(vector<Req>&pf, vector<int>&chunk, double&dev){
  int m=pf.size(); chunk.assign(m,0);
  long long W=0; int T=0;
  while(T<PF_TOK){
    int sel=-1;
    for(int i=0;i<m;i++) if(pf[i].wait>=AGE_CAP && chunk[i]==0 && pf[i].pf<pf[i].prompt){ sel=i; break; }  // age-cap spread
    if(sel<0){ double ideal=(double)(PF_WORK-W)/(PF_TOK-T); double bd=1e18;
      for(int i=0;i<m;i++){ if(pf[i].pf+chunk[i]>=pf[i].prompt) continue;
        double depth=pf[i].pf+chunk[i]+1; double x=fabs(depth-ideal); if(x<bd){bd=x;sel=i;} } }
    if(sel<0) break; chunk[sel]++; T++; W+=pf[sel].pf+chunk[sel];
  }
  dev = fabs((double)W-(double)PF_WORK)/PF_WORK;
  return T==PF_TOK && dev<=0.10;
}

int main(){
  int prefill=128; if(const char* e=getenv("PREFILL")) prefill=atoi(e);   // 배포 동작점 128 (env 로 256 등 스윕)
  double s=prefill/256.0; PF_TOK=prefill;                                 // 모든 스케일 값 prefill 에 선형
  DEC_CNT=(int)llround(123*s); DEC_KV=(long long)llround(12300000.0*s); PF_WORK=(long long)llround(25600000.0*s);
  DEC_POOL=2*DEC_CNT+10;   // 2 μ-batch 바닥(2×DEC_CNT) + 잉여 10 (Σdev plateau, 대형모델 위해 최소)
  PF_POOL =60;             // depth-diversity 하한 50 + 마진 10 (잉여 과대 = age-cap flood 위험)
  if(const char* e=getenv("DEC_POOL")) DEC_POOL=atoi(e);   // 잉여 스윕
  if(const char* e=getenv("PF_POOL"))  PF_POOL=atoi(e);
  if(const char* e=getenv("AGE_CAP"))  AGE_CAP=atoi(e);
  printf("통합 lifecycle | Z=%d, DEC_POOL=%d, PF_POOL=%d, AGE_CAP=%d. PULS 독립·composition 명중만.\n",
         Z,DEC_POOL,PF_POOL,AGE_CAP);
  printf("동작점 PREFILL=%d: 디코드(%d∧%.2fM)·프리필(%d∧%.2fM) 유지되나. (종속성·age-cap 포함)\n\n",
         PF_TOK, DEC_CNT, DEC_KV/1e6, PF_TOK, PF_WORK/1e6);

  vector<Node> nodes(Z);
  // 콜드스타트: 디코드 풀=센터 warm, 프리필 풀=센터·다양 depth
  for(auto&nd:nodes){
    for(int i=0;i<DEC_POOL;i++) admitDecodeCentered(nd,false);   // 콜드스타트: live KV 100K 센터(dec 랜덤)
    for(int i=0;i<PF_POOL;i++){ Req q; q.prompt=sampleB(); q.dtot=sampleDtot(q.prompt);   // 프리필 풀=wide·다양depth(게이트 X)
      q.pf=(int)(U()*q.prompt); q.dec=0; q.wait=0; nd.pf.push_back(q); }
  }

  const int ITERS=1000, WARM=500;
  double accDecHit=0,accPfHit=0,accDecDev=0,accPfDev=0,accDecMean=0,accDecPool=0,accReady=0; int N=0;
  double accDecPrompt=0,accDecTok=0,accN=0;   // DIAG
  double accPfPickDepth=0,accPfPromptM=0,accPfPfM=0,accPfDeep=0,accPfT=0,accPfTouched=0;   // DIAG
  long long transitions=0,completions=0;
  vector<int> picked, chunk; double dev;

  for(int it=0; it<ITERS; it++){
    for(auto&nd:nodes){
      // ── 프리필 steering ──
      bool ph=composePrefill(nd.pf,chunk,dev); double pfdev=dev;
      double pfW=0; long long pfT=0; double pfPromptM=0,pfPfM=0; int pfDeep=0;   // DIAG
      int pfTouched=0;   // DIAG: 이번 라운드 chunk 된 요청 수(=advance 폭, 디코드 246 대비)
      { for(int i=0;i<(int)nd.pf.size();i++){ if(chunk[i]>0){ pfW += (double)chunk[i]*nd.pf[i].pf + chunk[i]*(chunk[i]+1)/2.0; pfT+=chunk[i]; pfTouched++; }
          pfPromptM+=nd.pf[i].prompt; pfPfM+=nd.pf[i].pf; if(nd.pf[i].prompt>100000) pfDeep++; }
        int sz=nd.pf.size(); pfPromptM=sz?pfPromptM/sz:0; pfPfM=sz?pfPfM/sz:0; }
      double pfPickDepth = pfT?pfW/pfT:0;   // picked 토큰 평균 depth (타깃 100K)
      // 진행 + 전이(종속성): pf+=chunk; 완료→ready
      for(int i=0;i<(int)nd.pf.size();i++){ if(chunk[i]>0){ nd.pf[i].pf+=chunk[i]; nd.pf[i].wait=0; } else nd.pf[i].wait++; }
      { vector<Req> keep; for(auto&q:nd.pf){ if(q.pf>=q.prompt){ q.dec=0; q.wait=0; nd.ready.push_back(q); } else keep.push_back(q); }
        nd.pf.swap(keep); }
      // ── 디코드: 한 노드가 2 μ-batch 를 돌림 → 246 advance, 잉여 54 만 aging(age-cap 은 노드 내) ──
      vector<char> dused(nd.dec.size(),0);
      vector<int> p1,p2; double dv1=1,dv2=1;
      bool h1=composeDecodeBatch(nd.dec,dused,p1,dv1);
      bool h2=false; if((int)nd.dec.size()>=2*DEC_CNT) h2=composeDecodeBatch(nd.dec,dused,p2,dv2);
      double dh=(h1+h2)/2.0, dcdev=(dv1+dv2)/2.0;     // 두 배치 평균
      double dmean=0,pmean=0,tmean=0; { long long s=0,sp=0,st=0; for(auto&q:nd.dec){ s+=q.prompt+q.dec; sp+=q.prompt; st+=q.dec; }
        int sz=nd.dec.size(); dmean=sz?(double)s/sz:0; pmean=sz?(double)sp/sz:0; tmean=sz?(double)st/sz:0; }   // DIAG: footprint=prompt+dec 분리
      double navg=h2?((p1.size()+p2.size())/2.0):(double)p1.size();   // DIAG: 배치가 실제 채운 n
      // 진행: 선택분(2배치=246) dec++; 미선택(잉여) wait++; 완료 retire
      for(int i=0;i<(int)nd.dec.size();i++){ if(dused[i]){ nd.dec[i].dec++; nd.dec[i].wait=0; } else nd.dec[i].wait++; }
      { vector<Req> keep; for(auto&q:nd.dec){ if(q.dec>=q.dtot) completions++; else keep.push_back(q); } nd.dec.swap(keep); }
      // ── 디코드 풀 DEC_POOL 유지: ready(전이)로 먼저, 모자라면 힐링(센터 warm) ──
      while((int)nd.dec.size()<DEC_POOL && !nd.ready.empty()){ nd.dec.push_back(nd.ready.back()); nd.ready.pop_back(); transitions++; }
      while((int)nd.dec.size()<DEC_POOL) admitDecodeCentered(nd,true);   // 힐링: live KV 100K 센터(fresh dec=0)
      // ── 프리필 풀 PF_POOL 유지: fresh 센터(depth 0) admit ──
      while((int)nd.pf.size()<PF_POOL){ Req q; q.prompt=sampleB(); q.dtot=sampleDtot(q.prompt);   // 프리필 보충=wide
        q.pf=0; q.dec=0; q.wait=0; nd.pf.push_back(q); }
      // 측정
      if(it>=WARM){ accDecHit+=dh; accPfHit+=ph; accDecDev+=dcdev; accPfDev+=pfdev;
        accDecMean+=dmean; accDecPool+=nd.dec.size(); accReady+=nd.ready.size(); N++;
        accDecPrompt+=pmean; accDecTok+=tmean; accN+=navg;   // DIAG
        accPfPickDepth+=pfPickDepth; accPfPromptM+=pfPromptM; accPfPfM+=pfPfM; accPfDeep+=pfDeep; accPfT+=pfT; accPfTouched+=pfTouched; }   // DIAG
    }
  }
  printf("[steady-state, 마지막 %d라운드 × %d노드]\n", ITERS-WARM, Z);
  printf("디코드: 명중 %5.1f%%  Σ편차 %5.2f%%  풀평균kv %.0f  풀크기 %.0f\n",
    100.0*accDecHit/N, 100.0*accDecDev/N, accDecMean/N, accDecPool/N);
  printf("  └DIAG: prompt평균 %.0f + dec평균 %.0f = footprint %.0f (×%.2f)  | 배치 실제 n %.1f (타깃 %d)\n",
    accDecPrompt/N, accDecTok/N, (accDecPrompt+accDecTok)/N, (accDecPrompt+accDecTok)/accDecPrompt, accN/N, DEC_CNT);
  printf("프리필: 명중 %5.1f%%  Σ편차 %5.2f%%  (depth-work 타깃 %.2fM)\n",
    100.0*accPfHit/N, 100.0*accPfDev/N, PF_WORK/1e6);
  printf("  └DIAG: picked평균depth %.0f (타깃 100K)  | 풀 prompt평균 %.0f, pf평균 %.0f, prompt>100K %.0f  | T %.1f, chunk된요청 %.1f(=advance폭, 디코드 %d)\n",
    accPfPickDepth/N, accPfPromptM/N, accPfPfM/N, accPfDeep/N, accPfT/N, accPfTouched/N, 2*DEC_CNT);
  printf("종속성: 전이(프리필→디코드) %lld회, 디코드 완료 %lld회, ready 대기 %.1f\n",
    transitions, completions, accReady/N);
  printf("→ 종속성·age-cap 넣고도 두 composition 유지되면 통합 검증 성공.\n");
  return 0;
}
