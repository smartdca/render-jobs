# render one block of loop frames or warm-up clip frames to an mp4 (1920x1080, near-lossless)
# usage: python3 render_job.py loop <section> <i0> <i1> <out.mp4>
#        python3 render_job.py clip <warm-up#> <k0> <k1> <out.mp4>     (k = frame offset within the clip)
import sys, json, subprocess, os
import ar
W = json.load(open(os.path.join(ar.D, 'warmups.json')))
kind, a, b, c, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
enc = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', '1920x1080', '-r', '30', '-i', '-',
                        '-c:v', 'libx264', '-preset', 'medium', '-crf', '14', '-g', '30', '-pix_fmt', 'yuv420p', out], stdin=subprocess.PIPE)
if kind == 'loop':
    for i in range(b, min(c, ar.NL)): enc.stdin.write(ar.loop_frame(a, i).tobytes())
else:
    w = W[a]; f0 = int(round(w['t_a'] * 30)); f1 = int(round(w['t_b'] * 30))
    for k in range(b, min(c, f1 - f0)): enc.stdin.write(ar.clip_frame(W, w, f0 + k).tobytes())
enc.stdin.close(); enc.wait(); open(out + '.ok', 'w').write('ok'); print('ok', out)
