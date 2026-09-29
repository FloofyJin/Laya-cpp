#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <memory>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <netdb.h>
#include <sys/socket.h>
#include <unistd.h>

#include "laya.h"

namespace {

enum class Dir { Up, Down, Left, Right };
const Dir kAllDirs[4] = {Dir::Up, Dir::Down, Dir::Left, Dir::Right};

const char* dir_name(Dir d) {
    switch (d) {
        case Dir::Up:
            return "up";
        case Dir::Down:
            return "down";
        case Dir::Left:
            return "left";
        case Dir::Right:
            return "right";
    }
    return "";
}

bool dir_from_name(const std::string& name, Dir& out) {
    if (name == "up") {
        out = Dir::Up;
        return true;
    }
    if (name == "down") {
        out = Dir::Down;
        return true;
    }
    if (name == "left") {
        out = Dir::Left;
        return true;
    }
    if (name == "right") {
        out = Dir::Right;
        return true;
    }
    return false;
}

std::pair<int, int> delta(Dir d) {
    switch (d) {
        case Dir::Up:
            return {0, -1};
        case Dir::Down:
            return {0, 1};
        case Dir::Left:
            return {-1, 0};
        case Dir::Right:
            return {1, 0};
    }
    return {0, 0};
}

struct Point {
    int x;
    int y;
    bool operator==(const Point& o) const { return x == o.x && y == o.y; }
};

std::vector<Point> hamiltonian_cycle(int width, int height) {
    if (std::min(width, height) < 4 || (width % 2 != 0 && height % 2 != 0)) {
        throw std::runtime_error("board must be >= 4 in each dimension, with at least one even dimension");
    }
    if (height % 2 != 0) {
        const std::vector<Point> swapped = hamiltonian_cycle(height, width);
        std::vector<Point> out;
        out.reserve(swapped.size());
        for (const Point& p : swapped) {
            out.push_back({p.y, p.x});
        }
        return out;
    }
    std::vector<Point> path;
    path.push_back({0, 0});
    for (int y = 0; y < height; ++y) {
        if (y % 2 == 0) {
            for (int x = 1; x < width; ++x) {
                path.push_back({x, y});
            }
        } else {
            for (int x = width - 1; x >= 1; --x) {
                path.push_back({x, y});
            }
        }
    }
    for (int y = height - 1; y >= 1; --y) {
        path.push_back({0, y});
    }
    return path;
}

void verify_cycle(const std::vector<Point>& cycle, int width, int height) {
    const size_t capacity = static_cast<size_t>(width) * static_cast<size_t>(height);
    if (cycle.size() != capacity) {
        throw std::runtime_error("hamiltonian_cycle: wrong cell count");
    }
    std::vector<char> seen(capacity, 0);
    for (const Point& p : cycle) {
        if (p.x < 0 || p.x >= width || p.y < 0 || p.y >= height) {
            throw std::runtime_error("hamiltonian_cycle: cell out of bounds");
        }
        const size_t idx = static_cast<size_t>(p.y) * static_cast<size_t>(width) + static_cast<size_t>(p.x);
        if (seen[idx]) {
            throw std::runtime_error("hamiltonian_cycle: a cell was visited twice");
        }
        seen[idx] = 1;
    }
    for (size_t i = 0; i < cycle.size(); ++i) {
        const Point& a = cycle[i];
        const Point& b = cycle[(i + 1) % cycle.size()];
        if (std::abs(a.x - b.x) + std::abs(a.y - b.y) != 1) {
            throw std::runtime_error("hamiltonian_cycle: a step in the cycle is not to an adjacent cell");
        }
    }
}

struct Planner {
    int width;
    int height;
    std::vector<Point> cycle;
    std::vector<int> index_of;

    Planner(int w, int h) : width(w), height(h) {
        cycle = hamiltonian_cycle(w, h);
        verify_cycle(cycle, w, h);
        index_of.assign(static_cast<size_t>(w) * static_cast<size_t>(h), -1);
        for (size_t i = 0; i < cycle.size(); ++i) {
            index_of[static_cast<size_t>(cycle[i].y) * static_cast<size_t>(w) + static_cast<size_t>(cycle[i].x)] =
                static_cast<int>(i);
        }
    }

    int index_at(Point p) const {
        return index_of[static_cast<size_t>(p.y) * static_cast<size_t>(width) + static_cast<size_t>(p.x)];
    }

    int capacity() const { return width * height; }
};

struct MoveInfo {
    Dir direction;
    bool legal;
    bool safe;
    int advance;
    bool eats;
};

struct Game {
    int width;
    int height;
    std::deque<Point> body;
    Point food{0, 0};
    std::mt19937 rng;
    const Planner* planner;

    Game(int w, int h, unsigned seed, const Planner& pl, int initial_length = 3)
        : width(w), height(h), rng(seed), planner(&pl) {
        const Point start{w / 2, h / 2};
        const int start_idx = planner->index_at(start);
        const int cap = planner->capacity();
        for (int i = 0; i < initial_length; ++i) {
            const int idx = ((start_idx - i) % cap + cap) % cap;
            body.push_back(planner->cycle[static_cast<size_t>(idx)]);
        }
        place_food();
    }

    Point head() const { return body.front(); }

    bool occupied(Point p) const {
        for (const Point& s : body) {
            if (s == p) {
                return true;
            }
        }
        return false;
    }

    void place_food() {
        std::vector<Point> empty;
        for (const Point& p : planner->cycle) {
            if (!occupied(p)) {
                empty.push_back(p);
            }
        }
        if (empty.empty()) {
            food = body.front();
            return;
        }
        std::uniform_int_distribution<size_t> dist(0, empty.size() - 1);
        food = empty[dist(rng)];
    }

    bool in_bounds(Point p) const { return p.x >= 0 && p.x < width && p.y >= 0 && p.y < height; }

    Point target(Dir d) const {
        const auto [dx, dy] = delta(d);
        return {head().x + dx, head().y + dy};
    }

    bool is_legal(Dir d) const {
        const Point cell = target(d);
        if (!in_bounds(cell)) {
            return false;
        }
        if (body.size() > 1 && cell == body[1]) {
            return false;
        }
        const bool grows = (cell == food);
        for (size_t i = 0; i < body.size(); ++i) {
            if (!grows && i + 1 == body.size()) {
                continue;
            }
            if (body[i] == cell) {
                return false;
            }
        }
        return true;
    }

    std::vector<MoveInfo> moves() const {
        std::vector<MoveInfo> result;
        const int head_index = planner->index_at(head());
        const int cap = planner->capacity();
        const int tail_distance = ((planner->index_at(body.back()) - head_index) % cap + cap) % cap;
        const int food_distance = ((planner->index_at(food) - head_index) % cap + cap) % cap;
        for (Dir d : kAllDirs) {
            const bool legal = is_legal(d);
            const Point t = target(d);
            const int target_index = in_bounds(t) ? planner->index_at(t) : head_index;
            const int advance = ((target_index - head_index) % cap + cap) % cap;
            const bool eats = (t == food);
            bool safe = legal;
            if (safe && (advance > tail_distance || (advance == tail_distance && eats))) {
                safe = false;
            }
            if (safe && (advance == 0 || advance > food_distance)) {
                safe = false;
            }
            result.push_back({d, legal, safe, advance, eats});
        }
        return result;
    }

    std::pair<bool, int> food_reachability() const {
        const size_t cells = static_cast<size_t>(width) * static_cast<size_t>(height);
        std::vector<char> blocked(cells, 0);
        for (size_t i = 1; i < body.size(); ++i) {
            blocked[static_cast<size_t>(body[i].y) * static_cast<size_t>(width) + static_cast<size_t>(body[i].x)] =
                1;
        }
        std::vector<char> visited(cells, 0);
        std::deque<Point> queue;
        visited[static_cast<size_t>(head().y) * static_cast<size_t>(width) + static_cast<size_t>(head().x)] = 1;
        queue.push_back(head());
        int count = 1;
        while (!queue.empty()) {
            const Point p = queue.front();
            queue.pop_front();
            for (Dir d : kAllDirs) {
                const auto [dx, dy] = delta(d);
                const Point np{p.x + dx, p.y + dy};
                if (!in_bounds(np)) {
                    continue;
                }
                const size_t idx = static_cast<size_t>(np.y) * static_cast<size_t>(width) + static_cast<size_t>(np.x);
                if (blocked[idx] || visited[idx]) {
                    continue;
                }
                visited[idx] = 1;
                ++count;
                queue.push_back(np);
            }
        }
        const bool reachable =
            visited[static_cast<size_t>(food.y) * static_cast<size_t>(width) + static_cast<size_t>(food.x)] != 0;
        return {reachable, count};
    }
};

std::string move_label(const MoveInfo& m, bool has_preferred, Dir preferred) {
    if (!m.legal) {
        return "Blocked. Collision.";
    }
    if (!m.safe) {
        return "Unsafe. Traps the snake.";
    }
    if (m.eats) {
        return "Safe. Eat food now. Best.";
    }
    if (has_preferred && m.direction == preferred) {
        return "Safe. Best route to food.";
    }
    return "Safe. Slower route.";
}

std::string build_move_request(bool safe_route_exists, bool food_reachable, const std::vector<MoveInfo>& moves) {
    bool has_preferred = false;
    Dir preferred = Dir::Up;
    int best_advance = -1;
    for (const auto& m : moves) {
        if (m.safe && m.advance > best_advance) {
            best_advance = m.advance;
            preferred = m.direction;
            has_preferred = true;
        }
    }

    std::ostringstream ss;
    ss << "{\"state\":\"Safe route: " << (safe_route_exists ? "yes" : "no")
       << ". Food reachable through empty cells: " << (food_reachable ? "yes" : "no") << ".\",";
    ss << "\"questions\":{";
    ss << "\"move\":{\"type\":\"choice\",\"instructions\":\"Choose the best safe move toward food.\","
          "\"criteria\":{";
    for (size_t i = 0; i < moves.size(); ++i) {
        if (i) {
            ss << ",";
        }
        ss << "\"" << dir_name(moves[i].direction) << "\":\"" << move_label(moves[i], has_preferred, preferred)
           << "\"";
    }
    ss << "}},";
    ss << "\"risk\":{\"type\":\"noul\",\"instructions\":\"Is a safe route available?\"},";
    ss << "\"food\":{\"type\":\"noul\",\"instructions\":\"Is food reachable through empty cells?\"}";
    ss << "}}";
    return ss.str();
}

std::string extract_choice(const std::string& json) {
    const std::string key = "\"choice\":\"";
    const size_t pos = json.find(key);
    if (pos == std::string::npos) {
        return "";
    }
    const size_t start = pos + key.size();
    const size_t end = json.find('"', start);
    if (end == std::string::npos) {
        return "";
    }
    return json.substr(start, end - start);
}

std::string extract_probabilities_blob(const std::string& json) {
    const std::string key = "\"probabilities\":";
    const size_t pos = json.find(key);
    if (pos == std::string::npos) {
        return "{}";
    }
    const size_t start = pos + key.size();
    const size_t end = json.find('}', start);
    if (end == std::string::npos) {
        return "{}";
    }
    return json.substr(start, end - start + 1);
}

double extract_scoped_noul(const std::string& json, const std::string& key) {
    const std::string scope_key = "\"" + key + "\":{";
    const size_t scope_pos = json.find(scope_key);
    if (scope_pos == std::string::npos) {
        return 0.0;
    }
    const std::string noul_key = "\"noul\":";
    const size_t pos = json.find(noul_key, scope_pos);
    if (pos == std::string::npos) {
        return 0.0;
    }
    return std::strtod(json.c_str() + pos + noul_key.size(), nullptr);
}

double prob_for(const std::string& probabilities_blob, Dir d) {
    const std::string key = "\"" + std::string(dir_name(d)) + "\":";
    const size_t pos = probabilities_blob.find(key);
    if (pos == std::string::npos) {
        return -1.0;
    }
    return std::strtod(probabilities_blob.c_str() + pos + key.size(), nullptr);
}

Dir pick_best_by_probability(const std::vector<Dir>& options, const std::string& probabilities_blob) {
    Dir best = options.front();
    double best_p = -1.0;
    for (Dir d : options) {
        const double p = prob_for(probabilities_blob, d);
        if (p > best_p) {
            best_p = p;
            best = d;
        }
    }
    return best;
}

std::string http_post(const std::string& host, const std::string& port, const std::string& path,
                      const std::string& body) {
    struct addrinfo hints {};
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    struct addrinfo* res = nullptr;
    const int rc = getaddrinfo(host.c_str(), port.c_str(), &hints, &res);
    if (rc != 0) {
        throw std::runtime_error(std::string("getaddrinfo: ") + gai_strerror(rc));
    }

    int fd = -1;
    for (struct addrinfo* p = res; p != nullptr; p = p->ai_next) {
        fd = socket(p->ai_family, p->ai_socktype, p->ai_protocol);
        if (fd < 0) {
            continue;
        }
        if (connect(fd, p->ai_addr, p->ai_addrlen) == 0) {
            break;
        }
        close(fd);
        fd = -1;
    }
    freeaddrinfo(res);
    if (fd < 0) {
        throw std::runtime_error("could not connect to " + host + ":" + port);
    }

    std::ostringstream req;
    req << "POST " << path << " HTTP/1.1\r\n"
        << "Host: " << host << "\r\n"
        << "Content-Type: application/json\r\n"
        << "Content-Length: " << body.size() << "\r\n"
        << "Connection: close\r\n\r\n"
        << body;
    const std::string req_str = req.str();

    size_t sent = 0;
    while (sent < req_str.size()) {
        const ssize_t n = send(fd, req_str.data() + sent, req_str.size() - sent, 0);
        if (n <= 0) {
            close(fd);
            throw std::runtime_error("send() failed talking to " + host + ":" + port);
        }
        sent += static_cast<size_t>(n);
    }

    std::string response;
    char buf[4096];
    for (;;) {
        const ssize_t n = recv(fd, buf, sizeof(buf), 0);
        if (n < 0) {
            close(fd);
            throw std::runtime_error("recv() failed talking to " + host + ":" + port);
        }
        if (n == 0) {
            break;
        }
        response.append(buf, static_cast<size_t>(n));
    }
    close(fd);

    const size_t sep = response.find("\r\n\r\n");
    if (sep == std::string::npos) {
        throw std::runtime_error("malformed HTTP response from " + host + ":" + port);
    }
    const std::string status_line = response.substr(0, response.find("\r\n"));
    const std::string body_out = response.substr(sep + 4);
    if (status_line.find(" 200 ") == std::string::npos) {
        throw std::runtime_error("HTTP error from server: " + status_line + " " + body_out);
    }
    return body_out;
}

class Predictor {
public:
    virtual ~Predictor() = default;
    virtual std::string predict(const std::string& request_json) = 0;
};

class LocalPredictor : public Predictor {
public:
    LocalPredictor(const std::string& model_dir, const std::string& chip) {
        agent_ = laya_agent_load(model_dir.c_str(), chip.c_str(), err_, sizeof(err_));
        if (agent_ == nullptr) {
            throw std::runtime_error(err_);
        }
    }
    ~LocalPredictor() override {
        if (agent_ != nullptr) {
            laya_agent_free(agent_);
        }
    }
    std::string predict(const std::string& request_json) override {
        char* resp = laya_predict(agent_, request_json.c_str(), err_, sizeof(err_));
        if (resp == nullptr) {
            throw std::runtime_error(err_);
        }
        std::string out(resp);
        laya_free_string(resp);
        return out;
    }

private:
    laya_agent* agent_ = nullptr;
    char err_[256] = {};
};

class RemotePredictor : public Predictor {
public:
    RemotePredictor(std::string host, std::string port, std::string path)
        : host_(std::move(host)), port_(std::move(port)), path_(std::move(path)) {}

    std::string predict(const std::string& request_json) override {
        return http_post(host_, port_, path_, request_json);
    }

private:
    std::string host_;
    std::string port_;
    std::string path_;
};

struct Decision {
    Dir proposed;
    Dir executed;
    bool intervened;
    std::string probabilities;
    double risk_noul;
    double food_noul;
};

Decision decide(Predictor& predictor, const Game& game, bool guarded, bool debug_requests, int tick) {
    const std::vector<MoveInfo> moves = game.moves();
    std::vector<Dir> safe_dirs;
    for (const auto& m : moves) {
        if (m.safe) {
            safe_dirs.push_back(m.direction);
        }
    }
    const auto [reachable, open_cells] = game.food_reachability();
    (void)open_cells;

    const std::string request = build_move_request(!safe_dirs.empty(), reachable, moves);
    if (debug_requests) {
        std::fprintf(stderr, "[tick %d request] %s\n", tick, request.c_str());
    }
    const std::string response = predictor.predict(request);
    if (debug_requests) {
        std::fprintf(stderr, "[tick %d response] %s\n", tick, response.c_str());
    }

    Dir proposed;
    if (!dir_from_name(extract_choice(response), proposed)) {
        proposed = safe_dirs.empty() ? Dir::Up : safe_dirs.front();
    }
    const std::string probabilities = extract_probabilities_blob(response);

    Dir executed = proposed;
    bool intervened = false;
    if (guarded && !safe_dirs.empty() &&
        std::find(safe_dirs.begin(), safe_dirs.end(), proposed) == safe_dirs.end()) {
        executed = pick_best_by_probability(safe_dirs, probabilities);
        intervened = true;
    }

    return Decision{proposed, executed, intervened, probabilities, extract_scoped_noul(response, "risk"),
                    extract_scoped_noul(response, "food")};
}

void render(const Game& g, int tick, int eaten, int interventions, const std::string& verdict) {
    std::printf("\033[2J\033[H");
    std::vector<std::string> grid(static_cast<size_t>(g.height), std::string(static_cast<size_t>(g.width), '.'));
    for (size_t i = 0; i < g.body.size(); ++i) {
        grid[static_cast<size_t>(g.body[i].y)][static_cast<size_t>(g.body[i].x)] = (i == 0) ? 'O' : 'o';
    }
    grid[static_cast<size_t>(g.food.y)][static_cast<size_t>(g.food.x)] = '*';

    std::printf("+");
    for (int x = 0; x < g.width; ++x) {
        std::printf("-");
    }
    std::printf("+\n");
    for (const std::string& row : grid) {
        std::printf("|%s|\n", row.c_str());
    }
    std::printf("+");
    for (int x = 0; x < g.width; ++x) {
        std::printf("-");
    }
    std::printf("+\n");
    std::printf("tick=%d  length=%zu  food eaten=%d  interventions=%d\n", tick, g.body.size(), eaten,
               interventions);
    std::printf("%s\n", verdict.c_str());
}

int usage() {
    std::fprintf(stderr,
                 "usage: laya-snake --model DIR [--chip cpu|hip|cuda] [--width W] [--height H]\n"
                 "                  [--ticks N (0 = forever)] [--seed S]\n"
                 "       laya-snake --server HOST:PORT [--width W] [--height H]\n"
                 "                  [--ticks N (0 = forever)] [--seed S]\n");
    return 2;
}

}

int main(int argc, char** argv) {
    std::string model_dir = "models/laya";
    std::string chip = "cpu";
    std::string server;
    int width = 16;
    int height = 10;
    int max_ticks = 60;
    unsigned seed = 42;
    bool debug_requests = false;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--debug-requests") {
            debug_requests = true;
        } else if (arg == "--model" && i + 1 < argc) {
            model_dir = argv[++i];
        } else if (arg == "--server" && i + 1 < argc) {
            server = argv[++i];
        } else if (arg == "--ticks" && i + 1 < argc) {
            max_ticks = std::atoi(argv[++i]);
        } else if (arg == "--seed" && i + 1 < argc) {
            seed = static_cast<unsigned>(std::atoi(argv[++i]));
        } else if (arg == "--width" && i + 1 < argc) {
            width = std::atoi(argv[++i]);
        } else if (arg == "--height" && i + 1 < argc) {
            height = std::atoi(argv[++i]);
        } else if (arg == "--chip" && i + 1 < argc) {
            chip = argv[++i];
        } else {
            return usage();
        }
    }

    std::unique_ptr<Predictor> predictor;
    try {
        if (!server.empty()) {
            const size_t colon = server.find(':');
            if (colon == std::string::npos) {
                std::fprintf(stderr, "laya-snake: --server must be HOST:PORT, got '%s'\n", server.c_str());
                return 2;
            }
            predictor = std::make_unique<RemotePredictor>(server.substr(0, colon), server.substr(colon + 1),
                                                          "/predict");
        } else {
            predictor = std::make_unique<LocalPredictor>(model_dir, chip);
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "laya-snake: %s\n", e.what());
        return 1;
    }

    Planner planner(width, height);
    Game game(width, height, seed, planner);
    int eaten = 0;
    int interventions = 0;
    render(game, 0, eaten, interventions, "(starting)");

    for (int tick = 1; max_ticks <= 0 || tick <= max_ticks; ++tick) {
        const std::vector<MoveInfo> pre_moves = game.moves();
        const bool any_safe = std::any_of(pre_moves.begin(), pre_moves.end(), [](const MoveInfo& m) { return m.safe; });
        if (!any_safe) {
            render(game, tick, eaten, interventions, "no safe move available  ** TRAPPED **");
            std::printf("\nGame over at tick %d (trapped). length=%zu  food eaten=%d  interventions=%d\n", tick,
                       game.body.size(), eaten, interventions);
            return 0;
        }

        Decision decision;
        try {
            decision = decide(*predictor, game, true, debug_requests, tick);
        } catch (const std::exception& e) {
            render(game, tick, eaten, interventions, std::string("error: ") + e.what());
            return 1;
        }
        if (decision.intervened) {
            ++interventions;
        }

        const Dir chosen = decision.executed;
        const auto [dx, dy] = delta(chosen);
        const Point new_head{game.head().x + dx, game.head().y + dy};
        const bool eat = (new_head == game.food);

        char buf[256];
        std::snprintf(buf, sizeof(buf),
                     "laya proposed: %s, executed: %s%s\nrisk noul: %.4f  food noul: %.4f  probabilities: %s",
                     dir_name(decision.proposed), dir_name(decision.executed),
                     decision.intervened ? " (intervened)" : "", decision.risk_noul, decision.food_noul,
                     decision.probabilities.c_str());

        if (!game.in_bounds(new_head) || (!eat && new_head == game.body.back())) {
            render(game, tick, eaten, interventions, std::string(buf) + "\n** DIED **");
            std::printf("\nGame over at tick %d. length=%zu  food eaten=%d  interventions=%d\n", tick,
                       game.body.size(), eaten, interventions);
            return 0;
        }

        game.body.push_front(new_head);
        if (eat) {
            ++eaten;
            game.place_food();
        } else {
            game.body.pop_back();
        }

        render(game, tick, eaten, interventions, buf);
        std::this_thread::sleep_for(std::chrono::milliseconds(300));
    }

    std::printf("\nSurvived %d ticks. length=%zu  food eaten=%d  interventions=%d\n", max_ticks, game.body.size(),
               eaten, interventions);
    return 0;
}
