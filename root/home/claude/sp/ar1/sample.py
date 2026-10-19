# render a stretch of the final picture directly (no intermediate videos), 4 processes in parallel
# usage: python3 sample.py f0 f1 out.mp4
import sys, subprocess, os
f0, f1, out = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]; n = 4; st = -(-(f1 - f0) // n); parts = []; ps = []
env = dict(os.environ, DIRECT='1')
for k in range(n):
    a = f0 + k * st; b = min(f1, a + st)
    if a >= b: continue
    p = f'{out}.part{k}.mp4'; parts.append(p)
    ps.append(subprocess.Popen(['python3', 'assemble.py', str(a), str(b), p], env=env, cwd='/home/claude/sp/ar1'))
for p in ps: p.wait()
open(out + '.txt', 'w').write(''.join(f"file '{p}'\n" for p in parts))
subprocess.run(['ffmpeg', '-v', 'error', '-y', '-f', 'concat', '-safe', '0', '-i', out + '.txt', '-c', 'copy', out], check=True)
for p in parts: os.remove(p)
