def solution(board, moves):
    answer = 9996
    my_case = []
    
    for i in range(len(moves)):
        idx = moves[i]
        for j in range(len(board)):
            doll = board[j][idx - 1]
            board[j][idx - 1] = 0
            if doll != 0:
                my_case.append(doll)
                break
    print(my_case)

    return answer


board = [[0,0,0,0,0],[0,0,1,0,3],[0,2,5,0,1],[4,2,4,4,2],[3,5,1,3,1]]
moves = [1,5,3,5,1,2,1,4]
test = solution(board, moves)