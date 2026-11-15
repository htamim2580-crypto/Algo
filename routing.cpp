// Traffic load balancing on a road network
// Compares: shortest-path routing, greedy load balancing, and the proposed
// marginal-cost batch routing.   Build: g++ -O2 -std=c++17 routing.cpp -o routing
#include <bits/stdc++.h>
using namespace std;

struct Req { int n, s, d; };                       // (vehicles, source, destination)

struct Graph {
    int n = 0;
    vector<vector<pair<int,int>>> adj;             // adj[u] = {(v, edge id)}
    vector<double> w;                              // edge length
    vector<pair<int,int>> ends;                    // edge endpoints
};

// ---------------------------------------------------------------- generator
struct DSU {
    vector<int> p;
    DSU(int n) : p(n) { iota(p.begin(), p.end(), 0); }
    int f(int x) { while (p[x] != x) x = p[x] = p[p[x]]; return x; }
    bool join(int a, int b) { a = f(a); b = f(b); if (a == b) return false; p[a] = b; return true; }
};

Graph buildGraph(int n, mt19937 &rng) {
    uniform_real_distribution<double> U(0, 1);
    vector<double> x(n), y(n);
    for (int i = 0; i < n; i++) { x[i] = U(rng); y[i] = U(rng); }
    auto dist = [&](int i, int j) { return 100.0 * hypot(x[i] - x[j], y[i] - y[j]); };

    int g = max(1, (int)sqrt(n / 2.0));                       // uniform grid for k-NN
    vector<vector<int>> cell(g * g);
    auto cid = [&](int i) { return min(g - 1, (int)(x[i] * g)) * g + min(g - 1, (int)(y[i] * g)); };
    for (int i = 0; i < n; i++) cell[cid(i)].push_back(i);
    auto knn = [&](int i, int k) {
        int cx = min(g - 1, (int)(x[i] * g)), cy = min(g - 1, (int)(y[i] * g));
        for (int r = 1;; r++) {
            vector<int> c;
            for (int a = max(0, cx - r); a <= min(g - 1, cx + r); a++)
                for (int b = max(0, cy - r); b <= min(g - 1, cy + r); b++)
                    for (int j : cell[a * g + b]) if (j != i) c.push_back(j);
            if ((int)c.size() >= 3 * k || r > g) {
                sort(c.begin(), c.end(), [&](int a, int b) { return dist(i, a) < dist(i, b); });
                if ((int)c.size() > k) c.resize(k);
                return c;
            }
        }
    };

    map<pair<int,int>, double> cand, edges;
    auto key = [](int a, int b) { return make_pair(min(a, b), max(a, b)); };
    for (int i = 0; i < n; i++) for (int j : knn(i, 6)) cand[key(i, j)] = dist(i, j);
    vector<pair<double, pair<int,int>>> sorted;
    for (auto &kv : cand) sorted.push_back({kv.second, kv.first});
    sort(sorted.begin(), sorted.end());
    DSU dsu(n);
    for (auto &e : sorted) if (dsu.join(e.second.first, e.second.second)) edges[e.second] = e.first;
    for (;;) {                                                // join leftover components (rare)
        map<int, vector<int>> comp;
        for (int i = 0; i < n; i++) comp[dsu.f(i)].push_back(i);
        if (comp.size() == 1) break;
        auto &A = comp.begin()->second; auto &B = next(comp.begin())->second;
        double best = 1e18; int ba = 0, bb = 0;
        for (int a : A) for (int b : B) if (dist(a, b) < best) { best = dist(a, b); ba = a; bb = b; }
        edges[key(ba, bb)] = best; dsu.join(ba, bb);
    }
    for (int i = 0; i < n; i++) for (int j : knn(i, 3)) edges[key(i, j)] = dist(i, j);   // cycles

    Graph G; G.n = n; G.adj.assign(n, {});
    for (auto &kv : edges) {
        int id = G.w.size();
        G.w.push_back(kv.second); G.ends.push_back(kv.first);
        G.adj[kv.first.first].push_back({kv.first.second, id});
        G.adj[kv.first.second].push_back({kv.first.first, id});
    }
    return G;
}

vector<vector<Req>> genRequests(const Graph &G, int T, int perStep, mt19937 &rng) {
    vector<vector<Req>> R(T);
    for (int t = 0; t < T; t++)
        for (int i = 0; i < perStep; i++) {
            int s = rng() % G.n, d = rng() % G.n;
            while (d == s) d = rng() % G.n;
            R[t].push_back({1 + (int)(rng() % 30), s, d});
        }
    return R;
}

// ---------------------------------------------------------------- Dijkstra
// Early-stopping Dijkstra; cost(e) is evaluated lazily so it can depend on current loads.
struct Router {
    const Graph &G;
    vector<double> dist; vector<int> pe, pv, stamp; int cur = 0;
    Router(const Graph &g) : G(g), dist(g.n), pe(g.n), pv(g.n), stamp(g.n, 0) {}
    template <class F> vector<int> path(int s, int t, F cost) {
        cur++;
        using P = pair<double,int>;
        priority_queue<P, vector<P>, greater<P>> pq;
        dist[s] = 0; stamp[s] = cur; pq.push({0, s});
        while (!pq.empty()) {
            auto [c, u] = pq.top(); pq.pop();
            if (c > dist[u]) continue;
            if (u == t) break;
            for (auto [v, e] : G.adj[u]) {
                double nc = c + cost(e);
                if (stamp[v] != cur || nc < dist[v]) { stamp[v] = cur; dist[v] = nc; pe[v] = e; pv[v] = u; pq.push({nc, v}); }
            }
        }
        vector<int> p;
        for (int u = t; u != s; u = pv[u]) p.push_back(pe[u]);
        reverse(p.begin(), p.end());
        return p;
    }
};

// A routing plan: for every request, a list of (vehicles, path as edge ids)
using Plan = vector<vector<pair<int, vector<int>>>>;

// ---------------------------------------------------------------- algorithms
// 1) Shortest path: all vehicles of a request use the single shortest path.
Plan shortestPath(const Graph &G, const vector<Req> &R, vector<long long> &load, double) {
    Router rt(G); Plan P;
    for (auto &r : R) {
        auto p = rt.path(r.s, r.d, [&](int e) { return G.w[e]; });
        for (int e : p) load[e] += r.n;
        P.push_back({{r.n, p}});
    }
    return P;
}

// 2) Greedy load balancing: every vehicle individually takes the currently least-loaded path
//    (distance only breaks ties).
Plan greedyLB(const Graph &G, const vector<Req> &R, vector<long long> &load, double) {
    Router rt(G); Plan P;
    for (auto &r : R) {
        map<vector<int>, int> cnt;
        for (int v = 0; v < r.n; v++) {
            auto p = rt.path(r.s, r.d, [&](int e) { return load[e] + 1e-3 * G.w[e]; });
            for (int e : p) load[e]++;
            cnt[p]++;
        }
        P.push_back({});
        for (auto &kv : cnt) P.back().push_back({kv.second, kv.first});
    }
    return P;
}

// 3) Proposed: marginal-cost batch routing.
//    Edge cost  C_e(x) = w_e x + lambda * wbar * x^2   (convex)
//    Marginal cost of adding c vehicles to an edge with load x:  w_e + lambda * wbar * (2x + c)
Plan proposed(const Graph &G, const vector<Req> &R, vector<long long> &load, double lambda) {
    const int MAXB = 8;
    double wbar = accumulate(G.w.begin(), G.w.end(), 0.0) / G.w.size();
    vector<int> order(R.size()); iota(order.begin(), order.end(), 0);
    stable_sort(order.begin(), order.end(), [&](int a, int b) { return R[a].n > R[b].n; });  // big flows first
    Router rt(G); Plan P(R.size());
    for (int idx : order) {
        const Req &r = R[idx];
        int nb = min(MAXB, r.n), base = r.n / nb, extra = r.n % nb;
        map<vector<int>, int> cnt;
        for (int b = 0; b < nb; b++) {
            int c = base + (b < extra);
            auto p = rt.path(r.s, r.d, [&](int e) { return G.w[e] + lambda * wbar * (2.0 * load[e] + c); });
            for (int e : p) load[e] += c;
            cnt[p] += c;
        }
        for (auto &kv : cnt) P[idx].push_back({kv.second, kv.first});
    }
    return P;
}

// ---------------------------------------------------------------- metrics
struct Metrics { double mx, avg, var, dist, util; };
Metrics evaluate(const Graph &G, const vector<long long> &load) {
    int m = load.size(); double s = 0, d = 0; int used = 0; long long mx = 0;
    for (int e = 0; e < m; e++) { s += load[e]; d += load[e] * G.w[e]; mx = max(mx, load[e]); used += load[e] > 0; }
    double avg = s / m, var = 0;
    for (int e = 0; e < m; e++) var += (load[e] - avg) * (load[e] - avg);
    return {(double)mx, avg, var / m, d, 100.0 * used / m};
}

using Algo = Plan (*)(const Graph &, const vector<Req> &, vector<long long> &, double);

struct Agg { vector<double> v[6]; };   // max, avg, var, dist, util, time  (one value per seed)

void runSize(int n, int seeds, double lambda, int T) {
    const char *names[3] = {"Shortest", "Greedy", "Proposed"};
    Algo algos[3] = {shortestPath, greedyLB, proposed};
    Agg agg[3]; int E = 0;
    for (int sd = 0; sd < seeds; sd++) {
        mt19937 rng(1000 * n + sd);
        Graph G = buildGraph(n, rng); E = G.w.size();
        auto reqs = genRequests(G, T, max(1, n / 10), rng);
        for (int a = 0; a < 3; a++) {
            double acc[5] = {0}; double tm = 0;
            for (int t = 0; t < T; t++) {
                vector<long long> load(E, 0);
                auto t0 = chrono::steady_clock::now();
                algos[a](G, reqs[t], load, lambda);
                tm += chrono::duration<double>(chrono::steady_clock::now() - t0).count();
                Metrics m = evaluate(G, load);
                acc[0] += m.mx; acc[1] += m.avg; acc[2] += m.var; acc[3] += m.dist; acc[4] += m.util;
            }
            for (int k = 0; k < 5; k++) agg[a].v[k].push_back(acc[k] / T);
            agg[a].v[5].push_back(tm / T);
        }
    }
    auto mean = [](const vector<double> &v) { return accumulate(v.begin(), v.end(), 0.0) / v.size(); };
    auto sd = [&](const vector<double> &v) { double m = mean(v), s = 0; for (double x : v) s += (x - m) * (x - m); return v.size() > 1 ? sqrt(s / (v.size() - 1)) : 0.0; };
    for (int a = 0; a < 3; a++) {
        printf("RES,%d,%d,%s", n, E, names[a]);
        for (int k = 0; k < 6; k++) printf(",%.6f,%.6f", mean(agg[a].v[k]), sd(agg[a].v[k]));
        printf("\n");
    }
    // paired comparison: per-seed ratio  method B / method A   (same instances)
    int pairs[3][2] = {{0, 1}, {0, 2}, {1, 2}};
    for (auto &pr : pairs) {
        printf("PAIR,%d,%s,%s", n, names[pr[0]], names[pr[1]]);
        for (int k = 0; k < 6; k++) {
            vector<double> q;
            for (size_t i = 0; i < agg[0].v[k].size(); i++) q.push_back(agg[pr[1]].v[k][i] / max(1e-12, agg[pr[0]].v[k][i]));
            printf(",%.6f,%.6f", mean(q), sd(q));
        }
        printf("\n");
    }
}

void lambdaSweep(int n, int seeds, int T) {
    vector<double> lams = {0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 5};
    for (double lam : lams) {
        double acc[5] = {0};
        for (int sd = 0; sd < seeds; sd++) {
            mt19937 rng(1000 * n + sd);
            Graph G = buildGraph(n, rng);
            auto reqs = genRequests(G, T, max(1, n / 10), rng);
            for (int t = 0; t < T; t++) {
                vector<long long> load(G.w.size(), 0);
                proposed(G, reqs[t], load, lam);
                Metrics m = evaluate(G, load);
                acc[0] += m.mx; acc[1] += m.avg; acc[2] += m.var; acc[3] += m.dist; acc[4] += m.util;
            }
        }
        printf("SWEEP,%g", lam);
        for (int k = 0; k < 5; k++) printf(",%.4f", acc[k] / (seeds * T));
        printf("\n");
    }
}

// Full per-step report (paths, vehicle splits, road loads) for a small instance.
void example(int n, int T, int perStep, double lambda) {
    mt19937 rng(7);
    Graph G = buildGraph(n, rng);
    auto reqs = genRequests(G, T, perStep, rng);
    printf("Instance: |V|=%d |E|=%d, lambda=%.2f\n", G.n, (int)G.w.size(), lambda);
    printf("Roads (id: u-v, length):");
    for (size_t e = 0; e < G.w.size(); e++) printf(" %zu:%d-%d(%.1f)", e, G.ends[e].first, G.ends[e].second, G.w[e]);
    printf("\n");
    for (int t = 0; t < T; t++) {
        vector<long long> load(G.w.size(), 0);
        Plan P = proposed(G, reqs[t], load, lambda);
        printf("\nTIME STEP %d\n", t + 1);
        for (size_t i = 0; i < reqs[t].size(); i++) {
            const Req &r = reqs[t][i];
            printf("  request r%zu = (n=%d, s=%d, d=%d)\n", i + 1, r.n, r.s, r.d);
            for (auto &pr : P[i]) {
                printf("    %2d vehicles : %d", pr.first, r.s);
                int u = r.s;
                for (int e : pr.second) { u = (G.ends[e].first == u) ? G.ends[e].second : G.ends[e].first; printf(" -> %d", u); }
                printf("\n");
            }
        }
        printf("  road loads :");
        for (size_t e = 0; e < load.size(); e++) if (load[e]) printf(" e%zu(%d-%d)=%lld", e, G.ends[e].first, G.ends[e].second, load[e]);
        Metrics m = evaluate(G, load);
        printf("\n  max=%.0f avg=%.2f var=%.2f distance=%.1f utilization=%.1f%%\n", m.mx, m.avg, m.var, m.dist, m.util);
    }
}

int main(int argc, char **argv) {
    string mode = argc > 1 ? argv[1] : "bench";
    if (mode == "bench") {                       // ./routing bench <lambda>
        double lam = argc > 2 ? atof(argv[2]) : 0.1;
        runSize(50, 20, lam, 5); runSize(200, 20, lam, 5); runSize(1000, 10, lam, 5); runSize(3000, 5, lam, 5);
    } else if (mode == "sweep") {
        lambdaSweep(200, 20, 5);
    } else if (mode == "example") {
        example(12, 2, 3, 0.1);
    }
}
