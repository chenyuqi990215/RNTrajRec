// HMM_V4 初步断点调节 从基础版拖过来的
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <list>
#include <set>
#include <map>
#include <queue>

#define PI_Divided_by_180 0.0174532925199432957694
#define Length_Per_Rad 111226.29021121707545
#define INF 1e17
#define SIGMAZ 4.07
#define HLAF_SIGMAZ2 -0.0301843053
#define FRAC_SQR_2PI_SIGMAZ 0.0980202163
using namespace std;

////////////////Data structs///////////////////
class Point {
public
    double lon;
    double lat;
    int time;

    Point() : time(-1) {}
};

class Road {
public:
    vector<Point> path;
    int startPointId;
    int endPointId;
    int wayType;
    string wayName;
    double Len;
} r[50000];

class Traj {
public:
    vector<Point> path;
} t[150000];

class Score {
public:
    int RoadID;
    long double score;
    int preIndex;
    double distFromStart;
    Score() {}
    Score(int RoadID, long double score, int pre, double distLeft) {
        this->RoadID = RoadID;
        this->score = score;
        this->preIndex = pre;
        this->distFromStart = distLeft;
    }
};

class Node {
public:
    int end;
    double dist;

    bool operator<(const Node& rhs) const { return dist > rhs.dist; }
};

struct AdjNode {
    int endPointId;
    int edgeId;
    AdjNode* next;
};

///////////////////////////////////////////////////

///////////////////////The globals////////////////////
int n, m, k;
double maxLon, maxLat;
double minLon, minLat;
list<set<int>> Candidates;
list<int> ans[150000];
map<pair<int, int>, pair<double, double>> shortestDistPair;
int gridWidth, gridHeight;
double gridSize;
vector<AdjNode*> adjList;

//调参区域
list<int> grid[5000][5000];
int L = 50;
long double BETA[31] = { 0,           0.49037673,  0.82918373,  1.24364564,  1.67079581,  2.00719298,
                         2.42513007,  2.81248831,  3.15745473,  3.52645392,  4.09511775,  4.67319795,
                         5.41088180,  6.47666590,  6.29010734,  7.80752112,  8.09074504,  8.08550528,
                         9.09405065,  11.09090603, 11.87752824, 12.55107715, 15.82820829, 17.69496773,
                         18.07655652, 19.63438911, 25.40832185, 23.76001877, 28.43289797, 32.21683062,
                         34.56991141 };
//调参区域到此结束^_^!

////////////////////////////////////////////////////

////////////////////functions///////////////////

//点到点的距离
double distance(double lon1, double lat1, double lon2, double lat2) {
    double delta_lat = lat1 - lat2;
    double delta_lon = (lon2 - lon1) * cos(lat1 * PI_Divided_by_180);
    return Length_Per_Rad * sqrt(delta_lat * delta_lat + delta_lon * delta_lon);
}

//计算路径长度
double RoadLength(int i) {
    double length = 0;
    vector<Point>::iterator p, nxtp;
    p = nxtp = r[i].path.begin();
    nxtp++;
    while (nxtp != r[i].path.end()) {
        length += distance(p->lon, p->lat, nxtp->lon, nxtp->lat);
        p++;
        nxtp++;
    }
    return length;
}

void read() {
    scanf("%lf%lf%lf%lf",&minLat, &minLon, &maxLat, &maxLon);
    scanf("%d%d", &n, &k);
    //初始化AdjList
    for (int i = 0; i < k; ++i) {
        AdjNode* head = new AdjNode();
        head->endPointId = i;
        head->next = NULL;
        adjList.push_back(head);
    }
    // Read in roads
    for (int i = 0; i < n; ++i) {
        int id, p1, p2, c, way_type;
        char way_string[50];
        scanf("%d%d%d", &id, &p1, &p2);
        scanf("%s", &way_string);
        scanf("%d%d", &way_type, &c);
        r[id].startPointId = p1;
        r[id].endPointId = p2;
        r[id].wayName = way_string;
        r[id].wayType = way_type;

        // Read in adjList
        AdjNode* current = adjList[p1];
        while (current->next != NULL) current = current->next;
        AdjNode* tmpAdjNode = new AdjNode();
        tmpAdjNode->endPointId = p2;
        tmpAdjNode->edgeId = id;
        tmpAdjNode->next = NULL;
        current->next = tmpAdjNode;

        for (int j = 0; j < c; ++j) {
            double x, y;
            scanf("%lf%lf", &x, &y);
            Point* p = new Point();
            p->lat = x;
            p->lon = y;
            r[id].path.push_back(*p);
        }
        r[id].Len = RoadLength(id);
    }

    // Read in Trajs
    scanf("%d", &m);
    for (int i = 0; i < m; ++i) {
        while (1) {
            int time;
            scanf("%d", &time);
            if (time <= m)
                break;
            double x, y;
            scanf("%lf%lf", &x, &y);
            Point* p = new Point();
            p->lat = x;
            p->lon = y;
            p->time = time;
            t[i].path.push_back(*p);
        }
    }
}

double arccos(double lon, double lat, Point a, Point b) {
    double v1x = a.lon - lon;
    double v1y = a.lat - lat;
    double v2x = b.lon - a.lon;
    double v2y = b.lat - a.lat;
    return (v1x * v2x + v1y * v2y) / sqrt((v1x * v1x + v1y * v1y) * (v2x * v2x + v2y * v2y));
}

//点到边的距离
double distance(double lon, double lat, vector<Point>& path) {
    double ans = INF;
    vector<Point>::iterator p, nxtp;
    for (p = path.begin(); p != path.end(); ++p) {
        double tmp = distance(lon, lat, p->lon, p->lat);
        if (tmp < ans)
            ans = tmp;
    }
    p = path.begin();
    nxtp = path.begin();
    nxtp++;
    while (nxtp != path.end()) {
        if (arccos(lon, lat, *p, *nxtp) <= 0 && arccos(lon, lat, *nxtp, *p) <= 0) {
            double A = (nxtp->lat - p->lat);
            double B = -(nxtp->lon - p->lon);
            double C = p->lat * (nxtp->lon - p->lon) - p->lon * (nxtp->lat - p->lat);
            double tmp = abs(A * lon + B * lat + C) / sqrt(A * A + B * B) * Length_Per_Rad;
            if (tmp < ans)
                ans = tmp;
        }
        p++;
        nxtp++;
    }
    return ans;
}

//点到线距离，且返回距离起点的长度
double distance(double lon, double lat, vector<Point>& path, double& Len) {
    double tmpLen = 0;
    double ans = INF;
    double tmpd = 0;
    double x1 = -1, y1 = -1;
    for (vector<Point>::iterator Iter = path.begin(); Iter != path.end(); Iter++) {
        if (x1 != -1 && y1 != -1) {
            double x2 = Iter->lat;
            double y2 = Iter->lon;
            double dist = distance(y1, x1, lon, lat);
            if (dist < ans) {
                ans = dist;
                tmpLen = tmpd;
            }
            double v1x = x2 - x1;
            double v1y = y2 - y1;
            double v2x = lat - x1;
            double v2y = lon - y1;
            double v3x = lat - x2;
            double v3y = lon - y2;
            if (v1x * v2x + v1y * v2y > 0 && -v1x * v3x - v1y * v3y > 0 && (v1x != 0 || v1y != 0)) {
                double rate = ((lat - x2) * v1x + (lon - y2) * v1y) / (-v1x * v1x - v1y * v1y);
                double x = rate * x1 + (1 - rate) * x2;
                double y = rate * y1 + (1 - rate) * y2;
                double dist = distance(y, x, lon, lat);
                if (dist < ans) {
                    ans = dist;
                    tmpLen = tmpd + distance(y1, x1, y, x);
                }
            }
            tmpd += distance(y1, x1, y2, x2);
        }
        x1 = Iter->lat;
        y1 = Iter->lon;
    }
    Len = tmpLen;
    return ans;
}

bool cmp1(pair<double, double>& p1, pair<double, double>& p2) { return p1.first < p2.first; }

double EmissionPro(double distance) { return exp(distance * distance * HLAF_SIGMAZ2 * FRAC_SQR_2PI_SIGMAZ); }

double shortestPathLength(int Id1, int Id2, double deltaT) {
    vector<double> dist = vector<double>(k);
    vector<bool> flag = vector<bool>(k);
    for (int i = 0; i < k; ++i) {
        dist[i] = INF;
        flag[i] = false;
    }
    dist[Id1] = 0;
    priority_queue<Node> s;
    s.push((Node){ Id1, 0.0 });
    while (!s.empty()) {
        Node x = s.top();
        s.pop();
        int u = x.end;
        if (flag[u])
            continue;
        flag[u] = true;
        if (u == Id2)
            break;
        for (AdjNode* i = adjList[u]->next; i != NULL; i = i->next) {
            if (dist[i->endPointId] > dist[u] + r[i->edgeId].Len) {
                dist[i->endPointId] = dist[u] + r[i->edgeId].Len;
                s.push((Node){ i->endPointId, dist[i->endPointId] });
            }
        }
    }
    return dist[Id2];
}

int GetIndex(vector<Score>& row) {
    int res = -1;
    long double prob = -1;
    for (size_t i = 0; i < row.size(); i++) {
        if (prob < row[i].score) {
            prob = row.at(i).score;
            res = i;
        }
    }
    return res;
}

/////////////////////////////////////////////

////////GridIndex////////////

void Grid_Segment(int id, vector<Point>::iterator pt1, vector<Point>::iterator pt2) {
    double x1 = pt1->lon - minLon;
    double y1 = pt1->lat - minLat;
    double x2 = pt2->lon - minLon;
    double y2 = pt2->lat - minLat;
    int row = (int)(y1 / gridSize);
    int nxtrow = (int)(y2 / gridSize);
    int col = (int)(x1 / gridSize);
    int nxtcol = (int)(x2 / gridSize);
    int i;

    if (row == nxtrow && col == nxtcol) {
        if (row >= gridHeight || row < 0 || col >= gridWidth || col < 0 ||
            (grid[row][col].size() > 0 && grid[row][col].back() == id))
            return;
        grid[row][col].push_back(id);
        return;
    }

    if (row == nxtrow) {
        for (i = min(col, nxtcol); i <= max(col, col); ++i) {
            if (row >= gridHeight || row < 0 || col >= gridWidth || col < 0 ||
                (grid[row][col].size() > 0 && grid[row][col].back() == id))
                continue;
            grid[row][i].push_back(id);
        }
        return;
    }

    if (col == nxtcol) {
        for (i = min(row, nxtrow); i < max(row, nxtrow); ++i) {
            if (row >= gridHeight || row < 0 || col >= gridWidth || col < 0 ||
                (grid[row][col].size() > 0 && grid[row][col].back() == id))
                continue;
            grid[i][col].push_back(id);
        }
        return;
    }

    double A = y2 - y1;
    double B = x2 - x1;
    double C = x1 * y2 - x2 * y1;
    pair<double, double> plots[5000];
    int in = 0;
    for (i = min(row, nxtrow); i <= max(row, nxtrow); ++i) {
        plots[in++] = make_pair((C + B * i * gridSize) / A, i * gridSize);
    }
    for (i = min(col, nxtcol); i <= max(col, nxtcol); ++i) {
        plots[in++] = make_pair(i * gridSize, (-C + A * i * gridSize) / B);
    }
    plots[in++] = make_pair(x1, y1);
    plots[in++] = make_pair(x1, y1);
    sort(plots, plots + in, cmp1);

    for (i = 0; i < in - 1; ++i) {
        // always left-down
        int pts_row = (int)(plots[i].second / gridSize + 1e-9);
        int pts_col = (int)(plots[i].first / gridSize + 1e-9);
        int pts_nxtrow = (int)(plots[i + 1].second / gridSize + 1e-9);
        int pts_nxtcol = (int)(plots[i + 1].first / gridSize + 1e-9);
        int row = min(pts_row, pts_nxtrow);
        int col = min(pts_col, pts_nxtcol);
        if (row >= gridHeight || row < 0 || col >= gridWidth || col < 0 ||
            (grid[row][col].size() > 0 && grid[row][col].back() == id))
            continue;
        grid[row][col].push_back(id);
    }
    return;
}

void gridMaking() {
    gridWidth = int(distance(maxLon, minLat, minLon, minLat) / L) + 1;
    gridHeight = int((maxLat - minLat) / (maxLon - minLon) * double(gridWidth)) + 1;
    gridSize = (maxLon - minLon) / double(gridWidth);
    for (int i = 0; i < n; ++i) {
        vector<Point>::iterator p = r[i].path.begin();
        vector<Point>::iterator nxtp = r[i].path.begin();
        nxtp++;
        while (nxtp != r[i].path.end()) {
            Grid_Segment(i, p, nxtp);
            p++;
            nxtp++;
        }
    }
}
///////////////////////////////////////////////////

/////////////Algorithm/////////////

//第i条轨迹的候选
void getCandidate(int i) {
    Candidates.clear();
    for (auto j = t[i].path.begin(); j != t[i].path.end(); j++) {
        double x = j->lon - minLon;
        double y = j->lat - minLat;
        set<int> temp;
        set<int>::iterator q;
        int row = int(y / gridSize);
        int col = int(x / gridSize);
        for (int r = row - 1; r <= row + 1; ++r)
            for (int c = col - 1; c <= col + 1; ++c) {
                list<int>::iterator p;
                for (p = grid[r][c].begin(); p != grid[r][c].end(); p++) {
                    if (temp.find(*p) != temp.end())
                        continue;
                    temp.insert(*p);
                }
            }
        Candidates.push_back(temp);
    }
}

//隐马尔科夫
void matching_hmm() {
    for (int i = 0; i < m; ++i) {
        // 对于一条轨迹，获取每个采样点的候选路段
        // 整体思路如下：首先找到每个采样点所在的格子（每个格子大小是50m*50m的）
        // 所以如果只选择所载格子的所有路段会有遗漏
        // 根据论文中的做法，我们选取当前GPS所在的格子以及周围3*3区域的格子中所有路段作为候选路段
        getCandidate(i);
        list<set<int>>::iterator in = Candidates.begin();
        // 这句话获取这条轨迹平均采样率，用于得到转移概率中的Beta参数
        int sampleRate = (int)(t[i].path.size()) > 1 ? (t[i].path.back().time - t[i].path.front().time) /
                                                           ((int)(t[i].path.size()) - 1)
                                                     : (t[i].path.back().time - t[i].path.front().time);
        // 做一个越界的判断
        if (sampleRate > 30)
           sampleRate = 30;
        long double BT = BETA[sampleRate];
        vector<vector<Score>> scoreMatrix;
        vector<Point>::iterator formerTrajPoint = t[i].path.end();
        bool flag = true;
        int currentTrajPointIndex = 0;
        for (vector<Point>::iterator trajectoryIterator = t[i].path.begin();
             trajectoryIterator != t[i].path.end(); trajectoryIterator++) {
            // 定义相邻两个采样点之间的距离
            double distBetweenTwoTrajPoints;
            double deltaT = -1;
            if (formerTrajPoint != t[i].path.end()) {
                deltaT = trajectoryIterator->time - formerTrajPoint->time;
                distBetweenTwoTrajPoints = distance(trajectoryIterator->lon, trajectoryIterator->lat,
                                                    formerTrajPoint->lon, formerTrajPoint->lat);
            }
            long double currentMaxProb = -1e10;
            vector<Score> scores;
            long double* emissionProbs = new long double[in->size()];
            int currentCanadidateRoadIndex = 0;
            for (set<int>::iterator p = (*in).begin(); p != (*in).end(); p++) {
                int preColumnIndex = -1;
                double currentDistfromStart = 0;
                // 计算采样点到候选路段的距离
                double DistBetweenTrajPointAndRoad = distance(
                    trajectoryIterator->lon, trajectoryIterator->lat, r[*p].path, currentDistfromStart);
                emissionProbs[currentCanadidateRoadIndex] = EmissionPro(DistBetweenTrajPointAndRoad);
                // flag判定是不是第一个采样点
                if (!flag) {
                    long double currentMaxProbTmp = -1e10;
                    int formerCanadidateRoadIndex = 0;
                    // 遍历上一个采样点得到的Score矩阵
                    for (vector<Score>::iterator sp = scoreMatrix.back().begin();
                         sp != scoreMatrix.back().end(); sp++) {
                        double formerDistfromStart = sp->distFromStart;
                        double formerDistToEnd = r[sp->RoadID].Len - formerDistfromStart;
                        double routeNetworkDistBetweenTwoRoads;
                        double routeNetworkDistBetweenTwoTrajPoints;
                        if (*p == sp->RoadID)
                            // 如果上一个候选点和下一个候选点在同一个距离，可以直接求出距离
                            // 这是因为Score里面保存了这个GPS点到路段起始的距离
                            // 这个其实是一个时间的优化
                            routeNetworkDistBetweenTwoTrajPoints =
                                fabs(currentDistfromStart - formerDistfromStart);
                        else {
                            // 这个对最短路进行记忆化
                            // 就是说如果两个点计算过了，下一次就不用再算了
                            pair<int, int> ind = make_pair(r[sp->RoadID].endPointId, r[*p].startPointId);
                            // 找到了之前算过的最短路
                            if (shortestDistPair.find(ind) != shortestDistPair.end() &&
                                shortestDistPair[ind].first < INF)
                                routeNetworkDistBetweenTwoRoads = shortestDistPair[ind].first;
                            else {
                                routeNetworkDistBetweenTwoRoads = shortestPathLength(
                                        r[sp->RoadID].endPointId, r[*p].startPointId, deltaT);
                                shortestDistPair[ind] =
                                        make_pair(routeNetworkDistBetweenTwoRoads, deltaT);
                                }
                            }
                            routeNetworkDistBetweenTwoTrajPoints =
                                routeNetworkDistBetweenTwoRoads + currentDistfromStart + formerDistToEnd;
                        }
                        long double transactionProb =
                            exp(-fabs((long double)distBetweenTwoTrajPoints -
                                      (long double)routeNetworkDistBetweenTwoTrajPoints) /
                                BT) /
                            BT;
                        // 概率=转移概率*发射概率
                        long double tmpTotalProbForTransaction = sp->score * transactionProb;
                        if (currentMaxProbTmp < tmpTotalProbForTransaction) {
                            currentMaxProbTmp = tmpTotalProbForTransaction;
                            preColumnIndex = formerCanadidateRoadIndex;
                        }
                        formerCanadidateRoadIndex++;
                    }
                    emissionProbs[currentCanadidateRoadIndex] *= currentMaxProbTmp;
                }
                scores.push_back(Score(*p, emissionProbs[currentCanadidateRoadIndex], preColumnIndex,
                                       currentDistfromStart));
                // 记录当前最大的概率
                if (currentMaxProb < emissionProbs[currentCanadidateRoadIndex])
                    currentMaxProb = emissionProbs[currentCanadidateRoadIndex];
                currentCanadidateRoadIndex++;
            }
            delete[] emissionProbs;
            formerTrajPoint = trajectoryIterator;
            currentTrajPointIndex++;
            // 这里做了一个归一化，就是除了最大的概率
            // 这样可以保证计算过程中不会出现精度问题
            // 这是一个不错的优化！
            // 不会影响准确率，但是可以提高精度
            for (int j = 0; j < scores.size(); ++j) scores[j].score /= currentMaxProb;
            scoreMatrix.push_back(scores);
            // 如果没有候选点，那么就将轨迹断开。
            if (scores.size() == 0) {
                flag = true;
                formerTrajPoint = t[i].path.end();
            } else
                flag = false;
            in++;
            // cout << "第" << currentTrajPointIndex << "个候选路段匹配成功" << "\n";
        }
        // 这一段是从最后开始倒叙输出最后的答案
        int startColumnIndex = GetIndex(scoreMatrix.back());
        int temp = 0;
        for (int j = scoreMatrix.size() - 1; j >= 0; j--) {
            if (startColumnIndex != -1) {
                temp = scoreMatrix[j][startColumnIndex].RoadID;
                ans[i].push_front(temp);
                startColumnIndex = scoreMatrix[j][startColumnIndex].preIndex;
            } else  //断点处理
            {
                ans[i].push_front(temp);
                if (j > 0)
                    startColumnIndex = GetIndex(scoreMatrix[j - 1]);
            }
        }
        // cout << "第" << i << "条路正常" << "\n";
    }
}

//////////////////////////////////////////////////////////

int main() {
    read();
    gridMaking();
    matching_hmm();
    list<int>::iterator p;
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < t[i].path.size(); ++j) {
            cout << t[i].path[j].time << ' ' << t[i].path[j].lat << ' ' << t[i].path[j].lon << ' ' << ans[i].front() << endl;
            ans[i].pop_front();
        }
        cout << "-" << i + 1 << endl;
    }
    return 0;
}


void work(Trajectory t) {
    Trajectory new_t;
}