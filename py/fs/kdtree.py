import fs._fs as c

class KdTree:
    def __init__(self):
        self._list = c._kdtree_create_copy()
        self.height = c._kdtree_get_height()

    def __getitem__(self, idx):
        return self._list[idx]

    @staticmethod
    def left_child(i):
        return 2*i + 1

    @staticmethod
    def right_child(i):
        return 2*i + 2

