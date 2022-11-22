from utils.spatial_func import *
from utils.lru import LRUCache
from functools import lru_cache
MAX_ERR = 4000
LRU_CAPACITY = 20000

class SPSolver:
    def __init__(self, rn, use_ray=True, use_lru=True, max_err=4000):
        global MAX_ERR
        self.rn = rn
        self.use_ray = use_ray
        self.use_lru = use_lru
        self.sp_cache = LRUCache(capacity=LRU_CAPACITY)
        MAX_ERR = max_err

    @lru_cache(maxsize=LRU_CAPACITY, typed=True)
    def update_sp_cache(self, rx: int, ry: int) -> list:
        return \
            self.rn.shortestPath(rx, ry, with_route=True,
                                  max_len=MAX_ERR + self.rn.edgeDis[ry] + self.rn.edgeDis[rx])[1]

    def update_sp_lru(self, rx: int, ry: int) -> list:
        item = self.sp_cache.get((rx, ry))
        if item == [-1]:  # not hit:
            path = \
                self.rn.shortestPath(rx, ry, with_route=True,
                                      max_len=MAX_ERR + self.rn.edgeDis[ry] + self.rn.edgeDis[rx])[1]
            self.sp_cache.put((rx, ry), path)
            return path
        else:
            return item

    def update_sp_nolru(self, rx: int, ry: int) -> list:
        return self.rn.shortestPath(rx, ry, with_route=True,
                                      max_len=MAX_ERR + self.rn.edgeDis[ry] + self.rn.edgeDis[rx])[1]

    def sp(self, rx: int, ry: int) -> list:
        if not self.use_lru:
            return self.update_sp_nolru(rx, ry)
        elif self.use_ray:
            return self.update_sp_lru(rx, ry)
        else:
            return self.update_sp_cache(rx, ry)

    def cal_sp_dist(self, x: SPoint, y: SPoint, rx: int, ry: int) -> float:
        projection_x, rate_x, dist_x = project_pt_to_road(self.rn, x, rx)
        projection_y, rate_y, dist_y = project_pt_to_road(self.rn, y, ry)
        if rx == ry:
            return min(abs(rate_y - rate_x) * self.rn.edgeDis[rx], MAX_ERR)
        else:
            if not self.use_lru:
                travel_path_xy = self.update_sp_nolru(rx, ry)
                travel_path_yx = self.update_sp_nolru(ry, rx)
            elif self.use_ray:
                travel_path_xy = self.update_sp_lru(rx, ry)
                travel_path_yx = self.update_sp_lru(ry, rx)
            else:
                travel_path_xy = self.update_sp_cache(rx, ry)
                travel_path_yx = self.update_sp_cache(ry, rx)
            travel_dis_xy = (1 - rate_x) * self.rn.edgeDis[rx] + rate_y * self.rn.edgeDis[ry]
            travel_dis_yx = (1 - rate_y) * self.rn.edgeDis[ry] + rate_x * self.rn.edgeDis[rx]
            for rid in travel_path_xy[1:-1]:
                travel_dis_xy += self.rn.edgeDis[rid]
            for rid in travel_path_yx[1:-1]:
                travel_dis_yx += self.rn.edgeDis[rid]
            if travel_path_xy == [] or travel_dis_xy > MAX_ERR:
                travel_dis_xy = MAX_ERR
            if travel_path_yx == [] or travel_dis_yx > MAX_ERR:
                travel_dis_yx = MAX_ERR
            return min(travel_dis_xy, travel_dis_yx)