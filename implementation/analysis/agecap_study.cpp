// age-cap 필요성 측정 (PULS 독립, 기존 코드 미변경 — 별도 측정 파일).
//
// 질문: steering 단독(age-cap 없이)이면 off-size 요청이 계속 밀려(starve) 나는가?
//   노드 풀이 작고 매 배치가 풀의 ~40-50% 를 고르니, 안 밀릴 수도 있다 — 측정으로 판정.
// 측정: 디코드(123∧12.3M)·프리필(256∧25.6M-work) 각각, age-cap OFF vs ON 에서
//   ① 명중률·Σ편차(타깃 대비)  ② wait 분포(연속 미선택 라운드): 최대·평균·starve(>50)·미완료
//   추가: 디코드 "재선택"(177→123) 평균이 100K 서 얼마나 벗어나나(프리필 분석 포함).
//
// 결론(측정): age-cap OFF → composition 은 완벽(100%, 디코드 Σ편차 0.26%·프리필 0.03%)이나
//   off-size 요청이 영구 starve(wait=전체런 4000). age-cap ON → wait≤2, starve 0. ⇒ age-cap 필요.
//   ⚠ age-cap ON 의 명중률 저하는 본 study 의 pool/batch 비율·no-churn 탓 *과장*임 — age-cap 의
//   실제 composition 비용(spread)은 OPERATING_POINT §6.8/proto_steering 의 ~1-5%(AGE_CAP=2). 본
//   study 의 신뢰 결론은 wait/starvation(= age-cap 필요성)이지 composition 비용 magnitude 가 아니다.
//
// 빌드: g++ -O2 -std=c++17 analysis/agecap_study.cpp -o analysis/agecap_study.exe
#include <bits/stdc++.h>
using namespace std;

static mt19937 rng(11);
static inline double U(){ return uniform_real_distribution<double>(0.0,1.0)(rng); }
static inline int logU(double lo,double hi){ return (int)exp(U()*(log(hi)-log(lo))+log(lo)); }
static int sampleB(){ double u=U();
  if(u<0.20) return logU(1000,16000);
  if(u<0.90) return logU(16000,256000);
  return logU(256000,1000000); }
static int sampleDtot(int p){ return max(1000,(int)(p*(123.0/256.0)*(0.6+0.8*U()))); }
// gate: prompt 가 너무 길면(센터링 깨면) 버리고 다시 — 풀을 100K 센터 다양 풀로(엣지 격리 모사)
static int gatedPrompt(){ for(int t=0;t<1000;t++){ int p=sampleB(); if(p<=200000) return p; } return 100000; }

struct WaitStat { double hit, devPct, meanWait, p99Wait, forcedPct; int maxWait, starve; };
static void report(const char* tag,int agecap,const WaitStat&w){
  printf("%-8s age-cap %-3s | 명중 %5.1f%% Σ편차 %5.2f%% | wait 평균 %7.2f 최대 %5d | starve %7d | age-cap강제 %5.1f%%\n",
    tag, agecap>1000000?"OFF":"ON", w.hit, w.devPct, w.meanWait, w.maxWait, w.starve, w.forcedPct);
}

// ── DECODE: 풀 POOL, 매 라운드 steering 으로 123 선택(±age-cap). 선택분 dec++, 완료 retire→fresh.
static WaitStat studyDecode(int AGE_CAP){
  const int POOL=300, BATCH=123; const long long TGT=12300000LL;
  struct R{ int prompt,dtot,dec,wait,life; };
  vector<R> p(POOL);
  for(auto&r:p){ r.prompt=gatedPrompt(); r.dtot=sampleDtot(r.prompt); r.dec=(int)(U()*r.dtot); r.wait=0; r.life=0; }
  long long hitN=0,rounds=0; double devSum=0; long long waitObs=0,waitCnt=0; int maxWait=0,starve=0;
  long long forcedSum=0,pickSum=0;
  vector<int> waitHist;
  const int ROUNDS=4000, WARM=1000;
  for(int it=0; it<ROUNDS; it++){
    // steering: ideal=(TGT-S)/(BATCH-n) 최근접 + age-cap
    vector<char> used(POOL,0); long long S=0; int n=0; int forced=0;
    while(n<BATCH && S<TGT){
      int sel=-1; bool wasForced=false;
      for(int i=0;i<POOL;i++) if(!used[i] && p[i].wait>=AGE_CAP){ sel=i; wasForced=true; break; }   // age-cap 강제
      if(sel<0){ long long ideal=(long long)llround((double)(TGT-S)/(BATCH-n)); double bd=1e18;
        for(int i=0;i<POOL;i++) if(!used[i]){ double d=fabs((double)(p[i].prompt+p[i].dec)-ideal); if(d<bd){bd=d;sel=i;} } }
      if(sel<0) break; used[sel]=1; S+=p[sel].prompt+p[sel].dec; n++; if(wasForced)forced++;
    }
    if(it>=WARM){ forcedSum+=forced; pickSum+=n; }
    bool hit = (n==BATCH && fabs((double)S-(double)TGT)/TGT<=0.10);
    if(it>=WARM){ hitN+=hit; devSum+=fabs((double)S-(double)TGT)/TGT; rounds++; }
    // advance: 선택분 dec++, wait=0; 미선택 wait++; 완료 retire→fresh
    for(int i=0;i<POOL;i++){
      if(used[i]){ p[i].dec++; p[i].wait=0; }
      else { p[i].wait++; }
      p[i].life++;
      if(it>=WARM){ if(p[i].wait>maxWait)maxWait=p[i].wait; if(p[i].wait>50)starve++; waitObs+=p[i].wait; waitCnt++; if(p[i].wait>0)waitHist.push_back(p[i].wait); }
      if(p[i].dec>=p[i].dtot){ p[i].prompt=gatedPrompt(); p[i].dtot=sampleDtot(p[i].prompt); p[i].dec=0; p[i].wait=0; p[i].life=0; }
    }
  }
  sort(waitHist.begin(),waitHist.end());
  double p99 = waitHist.empty()?0:waitHist[(int)(waitHist.size()*0.99)];
  WaitStat w; w.hit=100.0*hitN/rounds; w.devPct=100.0*devSum/rounds;
  w.meanWait=waitCnt?(double)waitObs/waitCnt:0; w.p99Wait=p99; w.maxWait=maxWait; w.starve=starve;
  w.forcedPct = pickSum? 100.0*forcedSum/pickSum : 0;
  return w;
}

// ── PREFILL: 풀 POOL, 매 라운드 256 토큰 steering(±age-cap). depth=pf. 완료 pf==prompt→fresh.
static WaitStat studyPrefill(int AGE_CAP){
  const int POOL=512, TOK=256; const long long WORK=25600000LL;
  struct R{ int prompt,pf,wait,life; };
  vector<R> p(POOL);
  for(auto&r:p){ r.prompt=sampleB(); r.pf=(int)(U()*r.prompt); r.wait=0; r.life=0; }   // 다양 prompt·depth
  long long hitN=0,rounds=0; double devSum=0; long long waitObs=0,waitCnt=0; int maxWait=0,starve=0;
  long long forcedSum=0,pickSum=0;
  vector<int> waitHist;
  const int ROUNDS=4000, WARM=1000;
  for(int it=0; it<ROUNDS; it++){
    vector<int> chunk(POOL,0); long long W=0; int T=0; int forced=0;
    while(T<TOK){
      int sel=-1; bool wasForced=false;
      for(int i=0;i<POOL;i++) if(p[i].wait>=AGE_CAP && chunk[i]==0 && p[i].pf<p[i].prompt){ sel=i; wasForced=true; break; }  // spread: 요청당 1회
      if(sel<0){ double ideal=(double)(WORK-W)/(TOK-T); double bd=1e18;
        for(int i=0;i<POOL;i++){ if(p[i].pf+chunk[i]>=p[i].prompt)continue; double depth=p[i].pf+chunk[i]+1;
          double d=fabs(depth-ideal); if(d<bd){bd=d;sel=i;} } }
      if(sel<0) break; chunk[sel]++; T++; W+=p[sel].pf+chunk[sel]; if(wasForced)forced++;
    }
    if(it>=WARM){ forcedSum+=forced; pickSum+=T; }
    bool hit = (T==TOK && fabs((double)W-(double)WORK)/WORK<=0.10);
    if(it>=WARM){ hitN+=hit; devSum+=fabs((double)W-(double)WORK)/WORK; rounds++; }
    for(int i=0;i<POOL;i++){
      if(chunk[i]>0){ p[i].pf+=chunk[i]; p[i].wait=0; } else p[i].wait++;
      p[i].life++;
      if(it>=WARM){ if(p[i].wait>maxWait)maxWait=p[i].wait; if(p[i].wait>50)starve++; waitObs+=p[i].wait; waitCnt++; if(p[i].wait>0)waitHist.push_back(p[i].wait); }
      if(p[i].pf>=p[i].prompt){ p[i].prompt=sampleB(); p[i].pf=0; p[i].wait=0; p[i].life=0; }
    }
  }
  sort(waitHist.begin(),waitHist.end());
  double p99 = waitHist.empty()?0:waitHist[(int)(waitHist.size()*0.99)];
  WaitStat w; w.hit=100.0*hitN/rounds; w.devPct=100.0*devSum/rounds;
  w.meanWait=waitCnt?(double)waitObs/waitCnt:0; w.p99Wait=p99; w.maxWait=maxWait; w.starve=starve;
  w.forcedPct = pickSum? 100.0*forcedSum/pickSum : 0;
  return w;
}

int main(){
  printf("age-cap 필요성 측정 (steering 단독 OFF vs ON). 풀 abundant·다양. 4000라운드(warm 1000).\n");
  printf("wait=연속 미선택 라운드(=한 번도 안 뽑힌 라운드 수). starve=wait>50 관측수(off-size 영구 starve 지표).\n\n");
  const int OFF=1000000000, ON=2;
  report("decode",  OFF, studyDecode(OFF));
  report("decode",  ON,  studyDecode(ON));
  report("prefill", OFF, studyPrefill(OFF));
  report("prefill", ON,  studyPrefill(ON));
  return 0;
}
