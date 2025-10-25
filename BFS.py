from collections import deque

def bfs(n):
    visited = set()
    directions = [(1,0),(0,1),(0,-1),(-1,0)]
    start = (0,0)
    end = (n,n)
    queue = deque()
    queue.append([start,[(0,0)]])

    while queue:

        current_node , current_path = queue.popleft()
        visited.add(current_node)
        x,y = current_node

        if current_node == end:
            break

        for dx,dy in directions:
            new_x = x+dx
            new_y = y+dy
            if (new_x,new_y) not in visited:
                new_node = (new_x,new_y)
                new_path = current_path+[new_node]
                queue.append([new_node, new_path])



    return current_path
