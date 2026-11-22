#!/usr/bin/env python3
"""
Autonomous AI Agent for Kaggle (2x T4 + Qwen3.5 9B Vision + 128K Context)
Simpan sebagai /kaggle/working/agent.py lalu jalankan:
    python /kaggle/working/agent.py
"""

import os
import re
import sys
import glob
import time
import json
import base64
import shutil
import argparse
import subprocess
import urllib.request
from datetime import datetime, timezone, timedelta

DEFAULT_BASE_DIR = '/kaggle/working'
DEFAULT_PORT = 8080
DEFAULT_MAX_CTX = 131072

# =====================================================================
# 1. DEFINISI TOOLS OTONOM UNTUK AI AGENT
# =====================================================================
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "list_dir",
            "description": "Melihat daftar folder, file, dan jumlah baris kode di direktori mana pun (default proyek: /kaggle/working, bisa juga cek / atau folder sistem lain).",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path direktori target, misal /kaggle/working atau /"}
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Membaca isi file teks atau kode lengkap dengan nomor baris (mendukung hingga 7.500 baris sekaligus).",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path file yang akan dibaca"},
                    "start_line": {"type": "integer", "description": "Baris awal (opsional, default 1)"},
                    "end_line": {"type": "integer", "description": "Baris akhir (opsional, default 7500)"}
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "patch_file",
            "description": "Memperbaiki atau mengganti baris tertentu pada file besar tanpa menulis ulang seluruh file (otomatis membuat backup .bak).",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path file target"},
                    "start_line": {"type": "integer", "description": "Nomor baris awal yang diganti (1-indexed)"},
                    "end_line": {"type": "integer", "description": "Nomor baris akhir yang diganti (inklusif)"},
                    "new_code": {"type": "string", "description": "Kode pengganti yang sudah benar (tanpa nomor baris di depannya)"}
                },
                "required": ["path", "start_line", "end_line", "new_code"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Membuat file baru atau menimpa seluruh isi file kecil/menengah dengan konten lengkap.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path file tujuan"},
                    "content": {"type": "string", "description": "Isi file lengkap siap pakai"}
                },
                "required": ["path", "content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_bash",
            "description": "Menjalankan perintah shell/terminal non-interaktif di Kaggle (git, python, npm, pytest, curl, grep, dll) untuk cek error, build, atau verifikasi hasil kerja.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Perintah bash yang akan dieksekusi"},
                    "cwd": {"type": "string", "description": "Direktori kerja (opsional, default /kaggle/working)"}
                },
                "required": ["command"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "web_fetch",
            "description": "Mengambil isi halaman web, dokumentasi, atau API secara real-time dari URL untuk memverifikasi fakta/versi terbaru.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "URL lengkap diawali http:// atau https://"}
                },
                "required": ["url"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "inspect_image",
            "description": "Memuat file gambar dari disk (screenshot, UI, chart, foto) agar dapat dilihat dan dianalisis secara visual.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path file gambar (.png, .jpg, .webp)"},
                    "question": {"type": "string", "description": "Fokus hal yang ingin dicek pada gambar"}
                },
                "required": ["path"]
            }
        }
    }
]

PROJECT_IGNORE_DIRS = {'node_modules', '.git', '__pycache__', '.venv', 'venv', 'dist', 'build', '.next', '.cache'}
SYS_SHALLOW_DIRS = {'proc', 'sys', 'dev', 'usr', 'lib', 'lib64', 'var', 'etc', 'boot', 'run', 'opt', 'srv'}

HARD_PATTERNS = re.compile(
    r'\b(analisis|analisa|audit|debug|kenapa|error|bug|fix|benerin|perbaiki|algoritma|logika|'
    r'hitung|matematika|rancang|arsitektur|optimasi|bandingkan|buktikan|pecahkan|diagnosa|struktur|'
    r'strategi|riset|evaluasi|rencana|keputusan|pro kontra|skenario|dampak|studi kasus|cek|buatkan|bangun)\b',
    re.IGNORECASE
)


def encode_image_b64(img_path: str) -> str:
    ext = os.path.splitext(img_path)[1].lower().lstrip('.') or 'png'
    if ext == 'jpg':
        ext = 'jpeg'
    with open(img_path, 'rb') as f:
        b64 = base64.b64encode(f.read()).decode()
    return f'data:image/{ext};base64,{b64}'


def exec_tool(name: str, args: dict, base_dir: str = DEFAULT_BASE_DIR):
    try:
        if name == "list_dir":
            target = args.get("path", base_dir)
            if not os.path.isdir(target):
                return f"Error: Direktori '{target}' tidak ditemukan."
            out = [f"STRUKTUR DIREKTORI: {target}"]
            max_depth = 1 if os.path.abspath(target) == '/' else 3
            for root, dirs, files in os.walk(target):
                rel = os.path.relpath(root, target)
                depth = 0 if rel == '.' else rel.count(os.sep) + 1
                if depth >= max_depth:
                    dirs[:] = []
                else:
                    dirs[:] = sorted([
                        d for d in dirs
                        if d not in PROJECT_IGNORE_DIRS and (depth == 0 or d not in SYS_SHALLOW_DIRS)
                    ])

                indent = "  " * depth
                if rel != '.':
                    out.append(f"{'  ' * (depth - 1)}📁 {os.path.basename(root)}/")
                if depth == 0 and max_depth == 1:
                    for dname in sorted(dirs):
                        out.append(f"📁 {dname}/")
                    dirs[:] = []
                for fn in sorted(files)[:40]:
                    fp = os.path.join(root, fn)
                    try:
                        with open(fp, 'r', encoding='utf-8', errors='ignore') as f:
                            lc = sum(1 for _ in f)
                        out.append(f"{indent}📄 {fn} ({lc} baris)")
                    except Exception:
                        out.append(f"{indent}📄 {fn}")
                if len(out) > 120:
                    out.append("... [Daftar dipotong, gunakan path subfolder yang lebih spesifik]")
                    break
            return "\n".join(out)

        elif name == "read_file":
            fp = args["path"]
            if not os.path.exists(fp):
                return f"Error: File '{fp}' tidak ditemukan."
            s = max(1, int(args.get("start_line", 1)))
            e = int(args.get("end_line", 7500))
            with open(fp, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
            total = len(lines)
            e = min(total, e, s + 7499)
            sliced = ''.join(f"{i}: {lines[i-1]}" for i in range(s, e + 1))
            return f"FILE: {fp} (menampilkan baris {s}-{e} dari total {total} baris)\n\n{sliced}"

        elif name == "patch_file":
            fp = args["path"]
            if not os.path.exists(fp):
                return f"Error: File '{fp}' tidak ditemukan."
            s = int(args["start_line"])
            e = int(args["end_line"])
            new_code = args["new_code"]
            if new_code and not new_code.endswith('\n'):
                new_code += '\n'
            shutil.copy2(fp, fp + '.bak')
            with open(fp, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            if s < 1 or e > len(lines) or s > e:
                return f"Error: Rentang baris {s}-{e} tidak valid untuk file dengan {len(lines)} baris."
            lines[s - 1 : e] = new_code.splitlines(keepends=True)
            with open(fp, 'w', encoding='utf-8') as f:
                f.writelines(lines)
            return f"Sukses patch {fp} baris {s}-{e}. Total baris sekarang: {len(lines)} (backup di {fp}.bak)."

        elif name == "write_file":
            fp = args["path"]
            os.makedirs(os.path.dirname(os.path.abspath(fp)), exist_ok=True)
            with open(fp, 'w', encoding='utf-8') as f:
                f.write(args["content"])
            lc = args["content"].count('\n') + 1
            return f"Sukses menulis file {fp} ({lc} baris)."

        elif name == "run_bash":
            cmd_str = args["command"]
            cwd = args.get("cwd", base_dir)
            if not os.path.isdir(cwd):
                cwd = base_dir
            bash_env = os.environ.copy()
            bash_env['CI'] = 'true'
            bash_env['DEBIAN_FRONTEND'] = 'noninteractive'
            res = subprocess.run(
                cmd_str, shell=True, cwd=cwd, env=bash_env,
                stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=180
            )
            output = (res.stdout + "\n" + res.stderr).strip()
            if len(output) > 8000:
                output = output[:3500] + "\n\n... [OUTPUT DIPOTONG] ...\n\n" + output[-3500:]
            return f"EXIT CODE: {res.returncode}\nOUTPUT:\n{output or '(kosong)'}"

        elif name == "web_fetch":
            url = args["url"]
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=20) as r:
                raw = r.read().decode('utf-8', errors='ignore')
            clean = re.sub(r'<(script|style).*?>.*?</\1>', '', raw, flags=re.DOTALL | re.IGNORECASE)
            clean = re.sub(r'<[^>]+>', ' ', clean)
            clean = re.sub(r'\s+', ' ', clean).strip()
            return f"URL: {url}\nKONTEN:\n{clean[:7000]}"

        elif name == "inspect_image":
            fp = args["path"]
            if not os.path.exists(fp):
                return f"Error: Gambar '{fp}' tidak ditemukan."
            return {"__image_payload__": fp, "question": args.get("question", "Analisis gambar ini.")}

        return f"Error: Tool '{name}' tidak dikenal."
    except Exception as ex:
        return f"Error eksekusi {name}: {ex}"


# =====================================================================
# 2. REAL-TIME TELEMETRY, AUTO-PRUNING & ANTI-HALUSINASI
# =====================================================================
def get_now_wib() -> str:
    wib = timezone(timedelta(hours=7))
    now = datetime.now(wib)
    hari = ['Senin', 'Selasa', 'Rabu', 'Kamis', 'Jumat', 'Sabtu', 'Minggu'][now.weekday()]
    bulan = [
        '', 'Januari', 'Februari', 'Maret', 'April', 'Mei', 'Juni',
        'Juli', 'Agustus', 'September', 'Oktober', 'November', 'Desember'
    ][now.month]
    return f"{hari}, {now.day} {bulan} {now.year} pukul {now.strftime('%H:%M:%S')} WIB"


def get_realtime_telemetry(est_tokens: int, base_dir: str, max_ctx: int) -> str:
    try:
        total_d, _, free_d = shutil.disk_usage(base_dir)
        disk_info = f"Disk bebas: {free_d / (1024**3):.1f} GB / {total_d / (1024**3):.1f} GB"
    except Exception:
        disk_info = "Disk: N/A"

    try:
        gpu_out = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,noheader,nounits'],
            text=True, timeout=3
        ).strip().splitlines()
        gpu_info = " | ".join(
            f"GPU{i}: {line.split(',')[0].strip()}/{line.split(',')[1].strip()} MiB"
            for i, line in enumerate(gpu_out)
        )
    except Exception:
        gpu_info = "GPU: N/A"

    pct = (est_tokens / max_ctx) * 100
    return f"{disk_info} | {gpu_info} | Konteks terpakai: ~{pct:.1f}% ({est_tokens:,}/{max_ctx:,} token)"


def estimate_history_tokens(msgs: list) -> int:
    total_chars = 0
    for m in msgs:
        c = m.get('content')
        if isinstance(c, str):
            total_chars += len(c)
        elif isinstance(c, list):
            for part in c:
                if part.get('type') == 'text':
                    total_chars += len(part.get('text', ''))
                elif part.get('type') == 'image_url':
                    total_chars += 2000
    return total_chars // 3


def prune_old_tool_outputs(msgs: list, max_ctx: int, max_ratio: float = 0.78):
    while estimate_history_tokens(msgs) > int(max_ctx * max_ratio) and len(msgs) > 4:
        pruned = False
        for i in range(1, len(msgs) - 3):
            if msgs[i].get('role') == 'tool' and len(str(msgs[i].get('content', ''))) > 600:
                msgs[i]['content'] = "[Output tool lama diringkas otomatis untuk menghemat memori konteks]"
                pruned = True
                break
        if not pruned:
            del msgs[1]


def build_system_prompt(est_tokens: int, base_dir: str, max_ctx: int) -> str:
    return (
        "Kamu adalah Autonomous AI Agent & Partner Kerja Profesional berstandar global yang berjalan di lingkungan Kaggle.\n"
        f"WAKTU REAL-TIME: {get_now_wib()}.\n"
        f"STATUS SERVER REAL-TIME: {get_realtime_telemetry(est_tokens, base_dir, max_ctx)}.\n"
        f"WORKSPACE UTAMA: `{base_dir}` (dan `/kaggle/input` untuk dataset). Default pencarian proyek dimulai dari `{base_dir}`, "
        "namun kamu tetap bisa memeriksa `/` atau direktori sistem lain kapan pun diminta user.\n"
        "TOOLS TERSEDIA: `list_dir`, `read_file`, `patch_file`, `write_file`, `run_bash`, `web_fetch`, `inspect_image`.\n\n"
        "PRINSIP KERJA & ANTI-HALUSINASI WAJIB:\n"
        "1. KOMUNIKASI NATURAL & PROFESIONAL: Bicaralah layaknya rekan senior manusia yang santai, jelas, tajam, dan tidak kaku. "
        "Sebelum memanggil tool, beri kabar singkat apa yang sedang kamu lakukan.\n"
        "2. DILARANG KERAS HALUSINASI: Jangan pernah mengarang isi file, path, output terminal, atau informasi terkini. "
        "Gunakan `read_file`, `run_bash`, atau `web_fetch` untuk memverifikasi fakta secara real-time.\n"
        "3. BERTINDAK MANDIRI SAMPAI TUNTAS: Inspeksi file -> perbaiki dengan `patch_file` -> verifikasi lewat `run_bash` -> "
        "jika masih error, perbaiki lagi sampai benar-benar beres.\n"
        "4. PRESISI PATCH FILE BESAR: Selalu baca nomor baris dengan `read_file` sebelum memakai `patch_file`.\n"
        "5. KODE UTUH: Saat membuat file baru atau menulis kode di chat, tulis lengkap siap pakai tanpa placeholder."
    )


def auto_budget(text: str) -> int:
    return 2048 if HARD_PATTERNS.search(text) else 0


# =====================================================================
# 3. SERVER LIFECYCLE & AGENT LOOP
# =====================================================================
def start_llama_server(base_dir: str, port: int, max_ctx: int):
    srv_matches = glob.glob(f'{base_dir}/llama_bin/**/llama-server', recursive=True)
    if not srv_matches:
        raise FileNotFoundError(f"Binary llama-server tidak ditemukan di {base_dir}/llama_bin")
    srv = srv_matches[0]
    d = os.path.dirname(srv)

    model_matches = glob.glob(f'{base_dir}/model_gguf/*Q4_K_M.gguf')
    mmproj_matches = glob.glob(f'{base_dir}/model_gguf/*mmproj*.gguf')
    if not model_matches or not mmproj_matches:
        raise FileNotFoundError(f"File model GGUF atau mmproj tidak ditemukan di {base_dir}/model_gguf")

    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = '0,1'
    env['LD_LIBRARY_PATH'] = d + ':' + env.get('LD_LIBRARY_PATH', '')

    subprocess.run(['pkill', '-x', 'llama-server'], stderr=subprocess.DEVNULL)
    time.sleep(1)

    cmd = [
        srv, '-m', model_matches[0], '--mmproj', mmproj_matches[0],
        '-ngl', '99', '-c', str(max_ctx), '-np', '1',
        '--jinja', '--host', '127.0.0.1', '--port', str(port),
        '-fa', 'on', '-b', '2048', '-ub', '512', '-sm', 'layer',
        '--cache-type-k', 'q8_0', '--cache-type-v', 'q8_0'
    ]

    log_file = open('/tmp/server.log', 'w')
    proc = subprocess.Popen(cmd, env=env, stdout=log_file, stderr=subprocess.STDOUT)
    return proc, log_file


def wait_ready(proc, port: int, secs: int = 300) -> bool:
    health_url = f'http://127.0.0.1:{port}/health'
    for _ in range(secs // 2):
        if proc.poll() is not None:
            return False
        try:
            if urllib.request.urlopen(health_url, timeout=2).status == 200:
                return True
        except Exception:
            time.sleep(2)
    return False


def run_agent_turn(history: list, api_url: str, base_dir: str, max_ctx: int,
                   user_content, initial_budget: int = 0, max_steps: int = 15):
    history.append({'role': 'user', 'content': user_content})
    prune_old_tool_outputs(history, max_ctx)
    history[0]['content'] = build_system_prompt(estimate_history_tokens(history), base_dir, max_ctx)

    for step in range(max_steps):
        budget = initial_budget if step == 0 else 1024
        payload = {
            'messages': history,
            'tools': TOOLS,
            'tool_choice': 'auto',
            'max_tokens': -1,
            'temperature': 0.25,
            'stream': True,
            'cache_prompt': True,
            'reasoning_budget': budget,
            'chat_template_kwargs': {'enable_thinking': budget != 0},
        }
        req = urllib.request.Request(
            api_url,
            data=json.dumps(payload).encode(),
            headers={'Content-Type': 'application/json'}
        )

        ans_parts = []
        tool_calls_map = {}
        in_think = False
        started_text = False

        with urllib.request.urlopen(req, timeout=1800) as resp:
            for raw_line in resp:
                line = raw_line.decode('utf-8', errors='ignore').strip()
                if not line.startswith('data: '):
                    continue
                data_str = line[6:]
                if data_str == '[DONE]':
                    break
                chunk = json.loads(data_str)
                delta = chunk['choices'][0].get('delta', {})

                r_tok = delta.get('reasoning_content')
                if r_tok:
                    if not in_think:
                        print('\n💭 [Mikir...] ', end='', flush=True)
                        in_think = True
                    print(r_tok, end='', flush=True)

                c_tok = delta.get('content')
                if c_tok:
                    if in_think:
                        print('\n', flush=True)
                        in_think = False
                    if not started_text:
                        print('AI: ', end='', flush=True)
                        started_text = True
                    print(c_tok, end='', flush=True)
                    ans_parts.append(c_tok)

                for tc in delta.get('tool_calls') or []:
                    idx = tc.get('index', 0)
                    if idx not in tool_calls_map:
                        tool_calls_map[idx] = {
                            'id': tc.get('id') or f'call_{step}_{idx}',
                            'type': 'function',
                            'function': {'name': '', 'arguments': ''}
                        }
                    if tc.get('id'):
                        tool_calls_map[idx]['id'] = tc['id']
                    fn = tc.get('function') or {}
                    if fn.get('name'):
                        tool_calls_map[idx]['function']['name'] += fn['name']
                    if fn.get('arguments'):
                        tool_calls_map[idx]['function']['arguments'] += fn['arguments']

        if in_think:
            print('', flush=True)
        if started_text:
            print('', flush=True)

        ans_text = ''.join(ans_parts).strip()
        tool_calls = [tool_calls_map[i] for i in sorted(tool_calls_map.keys())]

        assistant_msg = {'role': 'assistant', 'content': ans_text or None}
        if tool_calls:
            assistant_msg['tool_calls'] = tool_calls
        history.append(assistant_msg)

        if not tool_calls:
            break

        pending_images = []
        for tc in tool_calls:
            fn_name = tc['function']['name']
            raw_args = tc['function']['arguments']
            try:
                parsed_args = json.loads(raw_args) if raw_args else {}
            except json.JSONDecodeError:
                parsed_args = {}

            preview = json.dumps(parsed_args, ensure_ascii=False)
            if len(preview) > 120:
                preview = preview[:117] + '...'
            print(f'⚙️ [Agent Action -> {fn_name}] {preview}', flush=True)

            result = exec_tool(fn_name, parsed_args, base_dir=base_dir)
            if isinstance(result, dict) and "__image_payload__" in result:
                img_fp = result["__image_payload__"]
                q_img = result["question"]
                history.append({
                    'role': 'tool',
                    'tool_call_id': tc['id'],
                    'name': fn_name,
                    'content': f"Gambar '{img_fp}' berhasil dimuat untuk inspeksi visual."
                })
                pending_images.append((img_fp, q_img))
            else:
                history.append({
                    'role': 'tool',
                    'tool_call_id': tc['id'],
                    'name': fn_name,
                    'content': str(result)
                })

        for img_fp, q_img in pending_images:
            history.append({
                'role': 'user',
                'content': [
                    {'type': 'text', 'text': f"[Inspeksi Visual: {img_fp}] {q_img}"},
                    {'type': 'image_url', 'image_url': {'url': encode_image_b64(img_fp)}}
                ]
            })

        prune_old_tool_outputs(history, max_ctx)
        history[0]['content'] = build_system_prompt(estimate_history_tokens(history), base_dir, max_ctx)


def main():
    parser = argparse.ArgumentParser(description="Autonomous AI Agent for Kaggle")
    parser.add_argument('--workdir', default=DEFAULT_BASE_DIR, help='Workspace directory (default: /kaggle/working)')
    parser.add_argument('--port', type=int, default=DEFAULT_PORT, help='Server port (default: 8080)')
    parser.add_argument('--ctx', type=int, default=DEFAULT_MAX_CTX, help='Max context tokens (default: 131072)')
    args, _ = parser.parse_known_args()

    base_dir = args.workdir
    port = args.port
    max_ctx = args.ctx
    api_url = f'http://127.0.0.1:{port}/v1/chat/completions'

    proc = None
    log_file = None
    history = [{'role': 'system', 'content': build_system_prompt(0, base_dir, max_ctx)}]

    try:
        print(f'Menyalakan Autonomous AI Agent ({get_now_wib()})...')
        proc, log_file = start_llama_server(base_dir, port, max_ctx)
        if not wait_ready(proc, port):
            raise RuntimeError('Server gagal jalan')

        print('Agent Siap! Telemetri real-time, Web Fetch, Vision, & Auto-Pruning aktif.')
        print('Opsional: /img <path_gambar> <pertanyaan> | /clear | exit\n' + '=' * 75)

        while True:
            user_in = input('\nLu: ').strip()
            if not user_in:
                continue
            if user_in.lower() in ('exit', 'quit'):
                break
            if user_in.lower() == '/clear':
                history.clear()
                history.append({'role': 'system', 'content': build_system_prompt(0, base_dir, max_ctx)})
                print('[Memori agent di-reset]')
                continue

            t0 = time.time()
            if user_in.startswith('/img '):
                parts = user_in[5:].strip().split(' ', 1)
                img_path = parts[0]
                prompt_txt = parts[1] if len(parts) > 1 else 'Jelaskan gambar ini dan lakukan tindakan jika diperlukan.'
                if not os.path.exists(img_path):
                    print(f'[File tidak ditemukan: {img_path}]')
                    continue
                content = [
                    {'type': 'text', 'text': prompt_txt},
                    {'type': 'image_url', 'image_url': {'url': encode_image_b64(img_path)}}
                ]
                run_agent_turn(history, api_url, base_dir, max_ctx, content, initial_budget=auto_budget(prompt_txt))
            else:
                run_agent_turn(history, api_url, base_dir, max_ctx, user_in, initial_budget=auto_budget(user_in))

            print(f'⏱️ Total waktu kerja: {time.time() - t0:.1f}s')

    except KeyboardInterrupt:
        print('\nDihentikan user.')
    except Exception as e:
        print('\nGAGAL:', e)
        if os.path.exists('/tmp/server.log'):
            with open('/tmp/server.log', 'r', errors='ignore') as lf:
                print(lf.read()[-1500:])
    finally:
        if proc and proc.poll() is None:
            proc.terminate()
            proc.wait()
        if log_file:
            log_file.close()
        print('Server dimatikan.')


if __name__ == '__main__':
    main()
