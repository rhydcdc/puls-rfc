// 클러스터 밸런싱 sim — 센터링 기반. PULS 독립.
//
// 설계(2026-06-03): 무한 풀(분포 B)에서 256 PULS 노드를 mean 100K·count 246~300 으로 채우고,
// 과도하게 긴 요청은 엣지로 격리. 동작점=노드 상주 풀이 (123,12.3M) compose 가능(on-point).
// avg 100K 는 노드 상주 풀 센터링 타깃이지 워크로드 강제 아님(OPERATING_POINT §4).
//
// 결론(cold-start 측정, cluster_balance.local.md): KK/LPT+swap 은 cap 때문에 단순 greedy 에
// 못 미쳐 폐기. cold-start = interleave greedy + 작은 E.
//
//  sim1 cold-start: Step1 gate(긴 것 shed, edge%=f(E)) + Step2 greedy(arrival순 min|mean-100K|).
//  sim3 healing   : steady-state churn(완료시 retire) → 전략적 greedy refill(빠진 만큼 풀에서
//                   ideal=(목표footprint−sum)/slot 최근접으로 당겨 count+mean 동시 복구).
//                   = 3-Phase(굵직한 독성→count펌핑→미세조합)의 통합형. inter-node swap 0, edge 0.
//
// 빌드: g++ -O2 -std=c++17 analysis/cluster_balance.cpp -o analysis/cluster_balance.exe
#include <bits/stdc++.h>
using namespace std;

static const int       Z        = 256;
static const long long CAP       = 30000000LL;
static const double    TGT_AVG   = 100000.0;
static const int       TGT_CNT   = 123;
static const long long TGT_KV    = 12300000LL;
static const int       NODE_MAX  = 300;
static const int       NODE_MIN  = 246;

static mt19937 rng(12345);
static inline double U(){ return uniform_real_distribution<double>(0.0,1.0)(rng); }
static inline int logU(double lo,double hi){ return (int)exp(U()*(log(hi)-log(lo))+log(lo)); }
static int sampleB(){ double u=U();        // short20%[1-16K]/mid70%[16-256K]/long10%[256K-1M]
  if(u<0.20) return logU(1000,16000);
  if(u<0.90) return logU(16000,256000);
  return logU(256000,1000000); }

struct Node { vector<int> v; long long sum=0;
  int cnt() const { return (int)v.size(); }
  double mean() const { return v.empty()?0.0:(double)sum/v.size(); }
};

// K개 disjoint (123,12.3M±10%) 배치를 몇 개나(0..K). steering closest-to-ideal, 실패 시 롤백·중단.
static int onpoint_k(const Node& nd, int K){
  int m=nd.cnt(); if(m<TGT_CNT) return 0;
  vector<long long> a(nd.v.begin(),nd.v.end()); sort(a.begin(),a.end());
  vector<char> used(m,0); int got=0;
  for(int b=0;b<K;b++){
    long long S=0; int n=0; vector<int> pick;
    while(n<TGT_CNT && S<TGT_KV){
      long long ideal=llround((double)(TGT_KV-S)/(TGT_CNT-n));
      int pos=(int)(lower_bound(a.begin(),a.end(),ideal)-a.begin());
      int Lp=pos-1,Rp=pos;
      while(Lp>=0&&used[Lp])Lp--; while(Rp<m&&used[Rp])Rp++;
      int best;
      if(Lp<0&&Rp>=m) break; else if(Lp<0) best=Rp; else if(Rp>=m) best=Lp;
      else best=((ideal-a[Lp])<=(a[Rp]-ideal))?Lp:Rp;
      used[best]=1; pick.push_back(best); S+=a[best]; n++;
    }
    if(n==TGT_CNT && fabs((double)S-(double)TGT_KV)/TGT_KV<=0.10) got++;
    else { for(int idx:pick) used[idx]=0; break; }
  }
  return got;
}

// K개 disjoint 배치를 compose 해 각 배치 Σkv 를 sums 에 담음(123 못 채우면 중단). 성공 개수 반환.
static int compose_batches(const Node& nd, int K, vector<long long>& sums){
  int m=nd.cnt(); if(m<TGT_CNT) return 0;
  vector<long long> a(nd.v.begin(),nd.v.end()); sort(a.begin(),a.end());
  vector<char> used(m,0); int got=0;
  for(int b=0;b<K;b++){
    long long S=0; int n=0; vector<int> pick;
    while(n<TGT_CNT && S<TGT_KV){
      long long ideal=llround((double)(TGT_KV-S)/(TGT_CNT-n));
      int pos=(int)(lower_bound(a.begin(),a.end(),ideal)-a.begin());
      int Lp=pos-1,Rp=pos;
      while(Lp>=0&&used[Lp])Lp--; while(Rp<m&&used[Rp])Rp++;
      int best;
      if(Lp<0&&Rp>=m) break; else if(Lp<0) best=Rp; else if(Rp>=m) best=Lp;
      else best=((ideal-a[Lp])<=(a[Rp]-ideal))?Lp:Rp;
      used[best]=1; pick.push_back(best); S+=a[best]; n++;
    }
    if(n==TGT_CNT){ sums.push_back(S); got++; }
    else { for(int idx:pick) used[idx]=0; break; }
  }
  return got;
}

// Step1: P개 draw → 긴 것부터 shed 해 kept-mean ≤ 100K+E. kept(asc), shed#, μ_s 반환.
static vector<int> gate(int P, double E, int& shed, double& mu){
  vector<int> w(P); for(auto&x:w) x=sampleB();
  sort(w.begin(),w.end(),greater<int>());
  long long sum=0; for(int x:w) sum+=x;
  int cnt=P,i=0; double ceil=TGT_AVG+E;
  while(cnt>0 && (double)sum/cnt > ceil){ sum-=w[i]; cnt--; i++; }
  shed=i; mu=cnt?(double)sum/cnt:0;
  vector<int> kept(w.begin()+i,w.end());
  return kept;
}

// Step2 greedy: arrival 순(shuffle), min|추가후mean-100K| (cnt<C, cap). 전량 배치, 못넣음=leftover.
static int place_greedy(vector<int> kept, int C, vector<Node>& nodes){
  shuffle(kept.begin(),kept.end(),rng); int leftover=0;
  for(int L:kept){
    int best=-1; double bd=1e18;
    for(int i=0;i<Z;i++){ const Node&nd=nodes[i];
      if(nd.cnt()>=C || nd.sum+L>CAP) continue;
      double d=fabs((double)(nd.sum+L)/(nd.cnt()+1)-TGT_AVG);
      if(d<bd){bd=d;best=i;} }
    if(best<0){leftover++;continue;}
    nodes[best].v.push_back(L); nodes[best].sum+=L;
  }
  return leftover;
}

// 무한 풀에서 ideal 에 가장 가까운(cap 적합) 원소 best-of-K 샘플. = 풀 일부만 쓰는 무한 stock.
static int pull_best(double ideal, long long capRoom, int K=200){
  long long best=-1; double bd=1e18;
  for(int k=0;k<K;k++){ int L=sampleB(); if(L>capRoom) continue;
    double d=fabs((double)L-ideal); if(d<bd){bd=d;best=L;} }
  return (int)best;
}

// 전략적 greedy healing: target 까지 ideal=(목표footprint−sum)/slot 최근접으로 풀에서 당김.
// ideal 큰 시작 = Phase1(굵직), 중간 = Phase2(count), 작은 끝 = Phase3(미세). 통합형. pulls 반환.
static int heal(Node& nd, int target){
  int pulls=0;
  while(nd.cnt()<target){
    int slots=target-nd.cnt();
    double ideal=(TGT_AVG*target - (double)nd.sum)/slots;
    if(ideal<1) ideal=1;
    long long capRoom=CAP-nd.sum;
    if(capRoom<=0) break;
    int L=pull_best(ideal, capRoom);
    if(L<0) break;
    nd.v.push_back(L); nd.sum+=L; pulls++;
  }
  return pulls;
}

struct Stat { double edge,cntMean,inRange,devAvg,devMax,onp1,onp2; int cntMin,edgeN; };
static Stat measure(vector<Node>& nodes, int edge, int P){
  Stat S{}; S.cntMin=INT_MAX; S.edgeN=edge;
  int inrange=0,o1=0,o2=0; double csum=0,dsum=0,dmax=0;
  for(auto&nd:nodes){ int c=nd.cnt(); S.cntMin=min(S.cntMin,c); csum+=c;
    if(c>=NODE_MIN&&c<=NODE_MAX) inrange++;
    double dv=fabs(nd.mean()-TGT_AVG); dsum+=dv; dmax=max(dmax,dv);
    int k=onpoint_k(nd,2); if(k>=1)o1++; if(k>=2)o2++; }
  S.edge=P?100.0*edge/(double)P:0; S.cntMean=csum/Z; S.inRange=100.0*inrange/Z;
  S.devAvg=dsum/Z; S.devMax=dmax; S.onp1=100.0*o1/Z; S.onp2=100.0*o2/Z;
  return S;
}

static vector<int> trim(vector<int> kept, int target){
  if((int)kept.size()<=target) return kept;
  shuffle(kept.begin(),kept.end(),rng); kept.resize(target); return kept;
}

int main(){
  const int P = 180000;
  printf("분포B short20/mid70/long10, Z=%d, cap30M. on-point=compose(123,12.3M±10%%).\n",Z);
  printf("on1=단일배치%%, on2=2 disjoint 배치%%(=floor 246 진짜 의미).\n\n");

  // ── sim1: cold-start greedy ──────────────────────────────────────────────
  printf("===== sim1: cold-start greedy (interleave) — E 스윕 =====\n");
  printf("%6s %6s %6s | %6s %7s %9s | %8s %8s | %6s %6s\n",
    "E(K)","edge#","edge%","Ccap","cntMin","∈246-300","|dev|avg","|dev|max","on1%","on2%");
  vector<double> Es={0,1000,2000,5000,10000,20000};
  for(double E:Es){
    int shed; double mu; vector<int> kept=gate(P,E,shed,mu);
    int C=min(NODE_MAX,(int)(CAP/mu)); vector<int> use=trim(kept,Z*C);
    vector<Node> nodes(Z); int lo=place_greedy(use,C,nodes);
    Stat S=measure(nodes,shed+lo,P);   // leftover 도 edge 로(못 받은 요청)
    printf("%6.0f %6d %6.2f | %6d %7d %8.1f%% | %8.0f %8.0f | %5.1f %5.1f\n",
      E/1000,S.edgeN,S.edge,C,S.cntMin,S.inRange,S.devAvg,S.devMax,S.onp1,S.onp2);
  }

  // ── sim3: steady-state churn + per-completion 힐링 ────────────────────────
  // cold-start(E=1K) 후, 매 라운드 각 상주가 완료확률 p 로 retire → 각 완료를 그 크기(ideal=hole)로
  // 보충(toxic-fit). edge 0(필요한 것만 당김). 시간에 따라 안정(drift 없음) 확인 = early vs late.
  printf("\n===== sim3: churn + per-completion 힐링 (E=1K cold-start, 완료확률 p, ROUNDS 라운드) =====\n");
  printf("%5s %6s | %6s %7s %9s | %8s %8s | %6s %6s %8s\n",
    "p%","window","cntMin","cntMean","∈246-300","|dev|avg","|dev|max","on1%","on2%","pulls/rd");
  vector<double> Ps={0.01,0.03,0.05};
  const int ROUNDS=300, WARM=150;
  for(double p:Ps){
    int shed; double mu; vector<int> kept=gate(P,1000.0,shed,mu);   // E=1K
    int C=min(NODE_MAX,(int)(CAP/mu)); vector<int> use=trim(kept,Z*C);
    vector<Node> nodes(Z); place_greedy(use,C,nodes);
    // 측정 누적: early(WARM 직후 1라운드) / late(마지막 WARM 라운드 평균)
    double lateInR=0,lateDevA=0,lateDevM=0,lateO1=0,lateO2=0,latePulls=0; int lateCntMin=INT_MAX; double lateCntMean=0; int lateN=0;
    double bSumDev=0,bMean=0; long long bN=0;             // 배치(123) 단위 통계
    Stat earlyS{};
    for(int rd=0; rd<ROUNDS; rd++){
      long long pulls=0;
      for(auto&nd:nodes){                                 // churn + per-completion 힐링
        vector<int> keep; long long s=0; vector<int> done;
        for(int L:nd.v){ if(U()>=p){keep.push_back(L);s+=L;} else done.push_back(L); }
        nd.v.swap(keep); nd.sum=s;
        for(int hole:done){                              // 각 완료를 그 크기(ideal=hole)로 보충 → toxic-fit
          long long cr=CAP-nd.sum; if(cr<=0)break;
          int L=pull_best((double)hole,cr); if(L<0)break;
          nd.v.push_back(L); nd.sum+=L; pulls++;
        }
      }
      if(rd==WARM){ earlyS=measure(nodes,0,0); }
      if(rd>=WARM){ Stat S=measure(nodes,0,0);
        lateInR+=S.inRange; lateDevA+=S.devAvg; lateDevM=max(lateDevM,S.devMax);
        lateO1+=S.onp1; lateO2+=S.onp2; lateCntMin=min(lateCntMin,S.cntMin);
        lateCntMean+=S.cntMean; latePulls+=pulls; lateN++;
        for(auto&nd:nodes){ vector<long long> sums;       // 실제 123-배치 Σ 측정
          compose_batches(nd,2,sums);
          for(long long S2:sums){ bSumDev+=fabs((double)S2-(double)TGT_KV)/TGT_KV;
            bMean+=(double)S2/TGT_CNT; bN++; } }
      }
    }
    printf("%5.0f %6s | %6d %7.1f %8.1f%% | %8.0f %8.0f | %5.1f %5.1f %8.0f\n",
      p*100,"early", earlyS.cntMin, earlyS.cntMean, earlyS.inRange, earlyS.devAvg,
      earlyS.devMax, earlyS.onp1, earlyS.onp2, 0.0);
    printf("%5s %6s | %6d %7.1f %8.1f%% | %8.0f %8.0f | %5.1f %5.1f %8.0f\n",
      "", "late", lateCntMin, lateCntMean/lateN, lateInR/lateN, lateDevA/lateN,
      lateDevM, lateO1/lateN, lateO2/lateN, latePulls/lateN);
    printf("%5s %6s | 실제 123-배치: 평균 %.0f토큰(타깃 100K), Σ|dev| %.2f%%(타깃 12.3M, ±10%% 밴드)\n",
      "", "batch", bMean/bN, 100.0*bSumDev/bN);
  }
  printf("(early=warmup 직후 1라운드, late=마지막 %d라운드 평균. drift 없으면 early≈late.)\n", ROUNDS-WARM);

  // ── sim3b: batched(평균) vs per-completion(hole 단위) — 긴 요청 starve 측정 ──
  // batched: 한 라운드 완료분 한꺼번에 빼고 ideal=평균 으로 보충 → toxic-fit 불가(평균에 뭉갬).
  // per-completion: 각 완료를 그 크기(ideal=hole)로 like-for-like 보충 → 긴 거 빠지면 긴 거 들어옴.
  printf("\n===== sim3b: healing 방식 — 긴 요청(≥256K) 보존 (E=1K, p=3%%, 300rd) =====\n");
  printf("%16s | %10s %10s | %10s | %6s %9s\n",
    "mode","long%cold","long%late","pull-long%","on2%","|dev|avg");
  const long long LONGTH=256000;
  for(int mode=0; mode<2; mode++){
    int shed; double mu; vector<int> kept=gate(P,1000.0,shed,mu);
    int C=min(NODE_MAX,(int)(CAP/mu)); vector<int> use=trim(kept,Z*C);
    vector<Node> nodes(Z); place_greedy(use,C,nodes);
    long long rc=0,rt=0; for(auto&nd:nodes) for(int L:nd.v){ rt++; if(L>=LONGTH)rc++; }
    double longCold=100.0*rc/rt;
    long long pullLong=0,pullTot=0; double p=0.03;
    for(int rd=0; rd<300; rd++) for(auto&nd:nodes){
      vector<int> keep; long long s=0; vector<int> done;
      for(int L:nd.v){ if(U()>=p){keep.push_back(L);s+=L;} else done.push_back(L); }
      nd.v.swap(keep); nd.sum=s;
      if(mode==0){                                          // batched: ideal=평균
        while(nd.cnt()<NODE_MAX){ int slots=NODE_MAX-nd.cnt();
          double ideal=(TGT_AVG*NODE_MAX-(double)nd.sum)/slots; if(ideal<1)ideal=1;
          long long cr=CAP-nd.sum; if(cr<=0)break; int L=pull_best(ideal,cr); if(L<0)break;
          nd.v.push_back(L); nd.sum+=L; pullTot++; if(L>=LONGTH)pullLong++; }
      } else {                                              // per-completion: ideal=hole
        for(int hole:done){ long long cr=CAP-nd.sum; if(cr<=0)break;
          int L=pull_best((double)hole,cr); if(L<0)break;
          nd.v.push_back(L); nd.sum+=L; pullTot++; if(L>=LONGTH)pullLong++; }
      }
    }
    long long rc2=0,rt2=0; for(auto&nd:nodes) for(int L:nd.v){ rt2++; if(L>=LONGTH)rc2++; }
    Stat S=measure(nodes,0,0);
    printf("%16s | %9.2f%% %9.2f%% | %9.2f%% | %5.1f %9.0f\n",
      mode==0?"batched(avg)":"per-completion", longCold, 100.0*rc2/rt2,
      100.0*pullLong/(pullTot>0?pullTot:1), S.onp2, S.devAvg);
  }
  printf("(long%%cold=cold-start 상주 긴요청 비율, long%%late=300rd 후. per-completion 이 보존하면 toxic-fit ✓)\n");
  printf("batch 줄 = 노드 상주 풀에서 실제 뽑은 123개 배치의 평균/Σ편차 — 300-평균이 아니라 이게 동작점.\n");
  return 0;
}
