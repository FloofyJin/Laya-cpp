#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "laya.h"

namespace {

enum class Dir { Up, Down, Left, Right };

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

Dir opposite(Dir d) {
    switch (d) {
        case Dir::Up:
            return Dir::Down;
        case Dir::Down:
            return Dir::Up;
        case Dir::Left:
            return Dir::Right;
        case Dir::Right:
            return Dir::Left;
    }
    return d;
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

int manhattan(Point a, Point b) {
    return std::abs(a.x - b.x) + std::abs(a.y - b.y);
}

struct Game {
    int width;
    int height;
    std::deque<Point> snake;
    Point food{0, 0};
    Dir last_dir = Dir::Right;
    std::mt19937 rng;

    Game(int w, int h, unsigned seed) : width(w), height(h), rng(seed) {
        snake.push_back({w / 2, h / 2});
        snake.push_back({w / 2 - 1, h / 2});
        snake.push_back({w / 2 - 2, h / 2});
        place_food();
    }

    bool occupied(Point p) const {
        for (const Point& s : snake) {
            if (s == p) {
                return true;
            }
        }
        return false;
    }

    void place_food() {
        std::uniform_int_distribution<int> dx(0, width - 1);
        std::uniform_int_distribution<int> dy(0, height - 1);
        Point p{0, 0};
        do {
            p = {dx(rng), dy(rng)};
        } while (occupied(p));
        food = p;
    }

    bool in_bounds(Point p) const { return p.x >= 0 && p.x < width && p.y >= 0 && p.y < height; }

    bool self_collision(Point p, bool about_to_eat) const {
        const size_t limit = about_to_eat ? snake.size() : snake.size() - 1;
        for (size_t i = 0; i < limit; ++i) {
            if (snake[i] == p) {
                return true;
            }
        }
        return false;
    }

    Point head() const { return snake.front(); }
};

std::string coord(Point p) {
    return "[" + std::to_string(p.x) + "," + std::to_string(p.y) + "]";
}

std::string build_request(const Game& g, Dir candidate) {
    const auto [dx, dy] = delta(candidate);
    const Point cand{g.head().x + dx, g.head().y + dy};

    std::ostringstream ss;
    ss << "{\"state\":{";
    ss << "\"board\":[" << g.width << "," << g.height << "],";
    ss << "\"snake\":[";
    for (size_t i = 0; i < g.snake.size(); ++i) {
        if (i) {
            ss << ",";
        }
        ss << coord(g.snake[i]);
    }
    ss << "],";
    ss << "\"food\":" << coord(g.food) << ",";
    ss << "\"direction\":\"" << dir_name(candidate) << "\",";
    ss << "\"candidate_head\":" << coord(cand);
    ss << "},";
    ss << "\"questions\":{\"unsafe\":{\"type\":\"noul\",";
    ss << "\"instructions\":\"The snake's head is about to move to candidate_head. Given board, "
          "snake and food, would this move make the snake collide with a wall or its own body?\"}}}";
    return ss.str();
}

double extract_noul(const std::string& json) {
    const std::string key = "\"noul\":";
    const size_t pos = json.find(key);
    if (pos == std::string::npos) {
        return 0.0;
    }
    return std::strtod(json.c_str() + pos + key.size(), nullptr);
}

void render(const Game& g, int tick, int eaten, int shields, const std::string& verdict) {
    std::printf("\033[2J\033[H");
    std::vector<std::string> grid(static_cast<size_t>(g.height), std::string(static_cast<size_t>(g.width), '.'));
    for (size_t i = 0; i < g.snake.size(); ++i) {
        grid[static_cast<size_t>(g.snake[i].y)][static_cast<size_t>(g.snake[i].x)] = (i == 0) ? 'O' : 'o';
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
    std::printf("tick=%d  length=%zu  food eaten=%d  shield interventions=%d\n", tick, g.snake.size(), eaten,
               shields);
    std::printf("laya: %s\n", verdict.c_str());
}

}

int main(int argc, char** argv) {
    std::string model_dir = "models/laya";
    std::string chip = "cpu";
    int width = 16;
    int height = 10;
    int max_ticks = 60;
    unsigned seed = 42;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--model" && i + 1 < argc) {
            model_dir = argv[++i];
        } else if (a == "--ticks" && i + 1 < argc) {
            max_ticks = std::atoi(argv[++i]);
        } else if (a == "--seed" && i + 1 < argc) {
            seed = static_cast<unsigned>(std::atoi(argv[++i]));
        } else if (a == "--width" && i + 1 < argc) {
            width = std::atoi(argv[++i]);
        } else if (a == "--height" && i + 1 < argc) {
            height = std::atoi(argv[++i]);
        } else if (a == "--chip" && i + 1 < argc) {
            chip = argv[++i];
        }
    }

    char err[256];
    laya_agent* agent = laya_agent_load(model_dir.c_str(), chip.c_str(), err, sizeof(err));
    if (agent == nullptr) {
        std::fprintf(stderr, "laya-snake: %s\n", err);
        return 1;
    }

    Game game(width, height, seed);
    int eaten = 0;
    int shields = 0;
    std::string verdict = "(starting)";
    render(game, 0, eaten, shields, verdict);

    for (int tick = 1; tick <= max_ticks; ++tick) {
        std::vector<Dir> candidates;
        for (Dir d : {Dir::Up, Dir::Down, Dir::Left, Dir::Right}) {
            if (d != opposite(game.last_dir)) {
                candidates.push_back(d);
            }
        }
        std::sort(candidates.begin(), candidates.end(), [&](Dir a, Dir b) {
            const auto [ax, ay] = delta(a);
            const auto [bx, by] = delta(b);
            const Point pa{game.head().x + ax, game.head().y + ay};
            const Point pb{game.head().x + bx, game.head().y + by};
            return manhattan(pa, game.food) < manhattan(pb, game.food);
        });

        Dir chosen = candidates.front();
        const std::string request = build_request(game, chosen);
        char* response = laya_predict(agent, request.c_str(), err, sizeof(err));
        if (response == nullptr) {
            verdict = std::string("error: ") + err;
            render(game, tick, eaten, shields, verdict);
            break;
        }
        const double p_unsafe = extract_noul(response);
        laya_free_string(response);

        bool shielded = false;
        if (p_unsafe > 0.5) {
            shielded = true;
            ++shields;
            bool found = false;
            for (Dir d : candidates) {
                const auto [dx, dy] = delta(d);
                const Point np{game.head().x + dx, game.head().y + dy};
                const bool eat = (np == game.food);
                if (game.in_bounds(np) && !game.self_collision(np, eat)) {
                    chosen = d;
                    found = true;
                    break;
                }
            }
            if (!found) {
                const Dir back = opposite(game.last_dir);
                const auto [dx, dy] = delta(back);
                const Point np{game.head().x + dx, game.head().y + dy};
                const bool eat = (np == game.food);
                if (game.in_bounds(np) && !game.self_collision(np, eat)) {
                    chosen = back;
                }
            }
        }

        const auto [dx, dy] = delta(chosen);
        const Point new_head{game.head().x + dx, game.head().y + dy};
        const bool eat = (new_head == game.food);

        char buf[128];
        std::snprintf(buf, sizeof(buf), "shield %s (p_unsafe=%.4f)%s",
                     p_unsafe > 0.5 ? "TRIGGERED" : "clear", p_unsafe, shielded ? " -> rerouted" : "");
        verdict = buf;

        if (!game.in_bounds(new_head) || game.self_collision(new_head, eat)) {
            render(game, tick, eaten, shields, verdict + "  ** DIED **");
            std::printf("\nGame over at tick %d. length=%zu  food eaten=%d  shield interventions=%d\n", tick,
                       game.snake.size(), eaten, shields);
            laya_agent_free(agent);
            return 0;
        }

        game.snake.push_front(new_head);
        if (eat) {
            ++eaten;
            game.place_food();
        } else {
            game.snake.pop_back();
        }
        game.last_dir = chosen;

        render(game, tick, eaten, shields, verdict);
        std::this_thread::sleep_for(std::chrono::milliseconds(300));
    }

    std::printf("\nSurvived %d ticks. length=%zu  food eaten=%d  shield interventions=%d\n", max_ticks,
               game.snake.size(), eaten, shields);
    laya_agent_free(agent);
    return 0;
}
