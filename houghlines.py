import os
import sys
import cv2
import numpy as np
import math
import pdb

def track_line(mask, x0, y0, x1, y1, max_gap, min_len):
#    pdb.set_trace()
    dx = np.absolute(x0 - x1)
    dy = np.absolute(y0 - y1)
    mx = np.minimum(x0,x1)
    my = np.minimum(y0,y1)
    lines = []
    if dx >= dy:
        k = (y0 - y1) / (x0 - x1)
        size = gap = 0
        startx = endx = 0
        starty = endy = 0
        for xinc in range(np.int64(dx)):
            x = xinc + mx
            y = y1 + k * (x - x1)
            if mask[y,x] > 0:
                gap = 0
                if size == 0: 
                    startx = x 
                    starty = y
                else:
                    endx = x 
                    endy = y
                size = size + 1
            else:
                if size == 0:
                    continue
                gap = gap + 1
                if gap < max_gap:
                    continue
                else:
                    if size >= min_len:
                        lines.append(startx)
                        lines.append(starty)
                        lines.append(endx)
                        lines.append(endy)
                    startx = starty = 0
                    endx = endy = 0
                    size = gap = 0
        if size > min_len:
            lines.append(startx)
            lines.append(starty)
            lines.append(endx)
            lines.append(endy)
    elif dy > dx:
        k = (x0 - x1) / (y0 - y1)
        size = gap = 0
        startx = endx = 0
        starty = endy = 0
        for yinc in range(np.int64(dy)):
            y = my + yinc
            x = x1 + k * (y - y1)
            if mask[y,x] > 0:
                gap = 0
                if size == 0: 
                    startx = x
                    starty = y
                else:
                    endx = x 
                    endy = y
                size = size + 1
            else:
                if size == 0:
                    continue
                gap = gap + 1
                if gap < max_gap:
                    continue
                else:
                    if size >= min_len:
                        lines.append(startx)
                        lines.append(starty)
                        lines.append(endx)
                        lines.append(endy)
                    startx = starty = 0
                    endx = endy = 0
                    size = gap = 0
        if size > min_len:
            lines.append(startx)
            lines.append(starty)
            lines.append(endx)
            lines.append(endy)
    if len(lines) == 0:
        return(np.array([]))
    cols = 4
    rows = len(lines) / cols;
    lines = np.array(lines)
    lines.shape = (rows,cols)
    return(lines)

#rho = x * cos(alpha) + y * sin(alpha)
def extract_line(mask, rho, alpha, max_gap, min_len):
    if np.absolute(rho) < 0.001:
        return(np.array([]))
    h,w = mask.shape
    sina = math.sin(alpha)
    cosa = math.cos(alpha)
    if np.absolute(sina) < 0.001:
        x0 = x1 = rho
        y0 = 0
        y1 = h - 1
    elif np.absolute(cosa) < 0.001:
        y0 = y1 = rho
        x0 = 0
        x1 = w - 1
    else:
        candis = np.zeros([4,3]) 

        candis[0,0] = 0
        candis[0,1] = rho / sina
        if candis[0,1] >= 0 and candis[0,1] < h:
            candis[0,2] = 1
       
        candis[1,0] = rho / cosa
        candis[1,1] = 0
        if candis[1,0] >= 0 and candis[1,0] < w:
            candis[1,2] = 1

        candis[2,0] = w - 1
        candis[2,1] = (rho - candis[2,0]*cosa) / sina
        if candis[2,1] >= 0 and candis[2,1] < h:
            candis[2,2] = 1


        candis[3,1] = h - 1
        candis[3,0] = (rho - candis[3,1]*sina) / cosa
        if candis[3,0] >= 0 and candis[3,0] < w:
            candis[3,2] = 1

        x0 = -1
        y0 = -1
        x1 = -1
        y1 = -1
        for k in range(4):
            if candis[k, 2] == 1:
                x0 = candis[k,0]
                y0 = candis[k,1]
                candis[k,2] = 0
                break
        for k in range(4):
            if candis[k, 2] == 1:
                x1 = candis[k,0]
                y1 = candis[k,1]
                candis[k,2] = 0
                break
        if x0 < 0 or x1 < 0 or y0 < 0 or y1 < 0:
            return(np.array([]))
        if x1 == x0 and y0 == y1:
            return(np.array([]))

    lines = track_line(mask,x0,y0,x1,y1, max_gap, min_len)
    return(lines)

def do_houghlines(mask, max_gap, min_len):
    htlines = cv2.HoughLines(mask, 1, np.pi/180, min_len)
    c,h,w = htlines.shape
    alllines = np.array([])
    #h = np.minimum(h,5000)
    for row in range(h):
        print("[%d/%d]\r\n" %(row+1, h))
        rho = htlines[0,row,0]
        alpha = htlines[0,row,1]
        lines = extract_line(mask, rho, alpha, max_gap, min_len)
        if lines.size > 0:
            if alllines.size > 1:
                alllines = np.vstack((alllines, lines))
            else:
                alllines = lines
    return(alllines)

if __name__ == "__main__":
    color = cv2.imread('d:\\dataset\\sample\\line1.jpg',1)
    if color.size < 1:
        print("can't load image\r\n")
    gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
    edge = cv2.Canny(gray, 10, 50)
    lines = do_houghlines(edge, 5, 30)
    h,w = lines.shape
    for row in range(h):
        color = cv2.cvtColor(edge, cv2.COLOR_GRAY2BGR)
        pt1 = (lines[row,0], lines[row,1])
        pt2 = (lines[row,2], lines[row,3])
        pt1 = tuple(np.int32(pt1))
        pt2 = tuple(np.int32(pt2))
        print("line %d %d %d %d\r\n" %(pt1[0],pt1[1], pt2[0], pt2[1]))
        if np.mod(row,2) == 0:
            cv2.line(color, pt1,pt2, (0,0,255))
        else:
            cv2.line(color, pt1,pt2, (0,255,0))
        outpath = ("d:\\tmp\\ht%d.bmp" %row)
        cv2.imwrite(outpath, color)

