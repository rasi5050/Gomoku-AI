import numpy as np
import os
from glob import glob
from tqdm import tqdm

'''
Dataset from https://gomocup.org/results/
'''
game_rule = 'Standard'  
base_path = '/Users/rasi/Documents/msCsSyracuse/sem3/courses/AI/Project/code/policies/CNN/gomocup2019results'

output_path = os.path.join('dataset', os.path.basename(base_path))
os.makedirs(output_path, exist_ok=True)

file_list = glob(os.path.join(base_path, '%s*/*.psq' % (game_rule, )))

for index, file_path in enumerate(tqdm(file_list)):
    with open(file_path, 'r') as f:
        lines = f.read().splitlines() 

    w, h = lines[0].split(' ')[1].strip(',').split('x')
    w, h = int(w), int(h)

    lines = lines[1:]

    inputs, outputs = [], []
    board = np.zeros([h, w], dtype=np.int8)

    

    # for i, line in enumerate(lines):
    for i, (line, line2) in enumerate(zip(lines, lines[1:])):
        if ',' not in line or ',' not in line2:
            break

        x, y, t = np.array(line.split(','), np.int8)
        x2, y2, t2 = np.array(line2.split(','), np.int8)

        #converting to zero based index
        x,y,x2,y2 = x-1,y-1,x2-1,y2-1

        input = board.copy().astype(np.int8)

        if i % 2 == 0:
            #max player
            player = 1
        else:
            #min player
            player = -1
        input[y][x] = player

        # output = np.zeros([h, w], dtype=np.int8)
        output = (y2, x2)



        # augmentation
        # rotate 4 x flip 3 = 12

        x2,y2=y2,x2


        # rotate x0
        inputs.append(input)
        outputs.append(output)
        inputs.append(np.fliplr(input))
        outputs.append((x2, 14-y2))
        inputs.append(np.flipud(input))
        outputs.append((14-x2, y2))

        # rotate x1
        inputs.append(np.rot90(input, k=1))
        outputs.append((y2, 14 - x2))
        inputs.append(np.fliplr(np.rot90(input, k=1)))
        outputs.append((y2, x2))
        inputs.append(np.flipud(np.rot90(input, k=1)))
        outputs.append((14 - y2, 14 - x2))

        # rotate x2
        inputs.append(np.rot90(input, k=2))
        outputs.append((14 - x2, 14 - y2))
        inputs.append(np.fliplr(np.rot90(input, k=2)))
        outputs.append((14 - x2, y2))
        inputs.append(np.flipud(np.rot90(input, k=2)))
        outputs.append((x2, 14 - y2))

        # rotate x3
        inputs.append(np.rot90(input, k=3))
        outputs.append((14 - y2, x2))
        inputs.append(np.fliplr(np.rot90(input, k=3)))
        outputs.append((14 - y2, 14 - x2))
        inputs.append(np.flipud(np.rot90(input, k=3)))
        outputs.append((y2, x2))
        
        # update board
        board[y][x] = player
        # break

    # break
    
    # save dataset
    np.savez_compressed(os.path.join(output_path, '%s.npz' % (str(index).zfill(5))), inputs=inputs, outputs=outputs)

print(inputs, outputs)
print(lines[-2], lines[-1])
print(file_path)