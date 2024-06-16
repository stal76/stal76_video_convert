#!/usr/bin/python3
# stal76 video converter - version 1.2.0

# https://docs.nvidia.com/video-technologies/video-codec-sdk/12.0/ffmpeg-with-nvidia-gpu/index.html
# https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=deb_local

import os
import os.path
import subprocess
import sys
import threading
import time

extensions_video = ['mp4', 'mov', 'avi', 'mpg', 'thm', 'mts', 'vob', 'webm', 'mod', 'gif']
extensions_not_video = ['jpg', 'jpeg', 'png', 'rar', 'txt', 'rtf', 'mp3', 'doc', 'xcf', 'odt']

def how_usage():
    print(f'Usage:\t{sys.argv[0]} source_dir dest_dir threads_count [cuda]')

def create_directory(dir_name):
    if os.path.exists(dir_name):
        return
    create_directory(os.path.dirname(dir_name))
    os.mkdir(dir_name)

def work(src_dir, dest_dir, threads_count, use_cuda):
    for dir_name in [src_dir, dest_dir]:
        if not os.path.exists(dir_name):
            print(f'Not found dir: {dir_name}')
            return
    threads_count = int(threads_count)

    # Build work file list
    all_files = []
    bad_extensions = set()
    for (dirpath, _, filenames) in os.walk(src_dir):
        for fname in filenames:
            ext = fname.split('.')[-1].lower()
            if ext in extensions_video:
                all_files.append((dirpath, fname))
            elif ext not in extensions_not_video:
                bad_extensions.add(ext)
    if len(bad_extensions) != 0:
        print(f'Found unknown extensions: {sorted(list(bad_extensions))}')
    work_list = []
    for dirpath, fname in all_files:
        src_name = os.path.join(dirpath, fname)
        dst_name = src_name[len(src_dir):]
        if dst_name[0] in ['/', '\\']:
            dst_name = dst_name[1:]
        dst_name = os.path.join(dest_dir, dst_name)
        if not os.path.exists(dst_name):
            work_list.append((src_name, dst_name))
            # Create directory
            dir_name = os.path.dirname(dst_name)
            if not os.path.exists(dir_name):
                print(f'Create dir: {dir_name}')
                create_directory(dir_name)
    work_list = sorted(work_list)
    # print(work_list)

    if len(work_list) == 0:
        print('Not found files')
        return
    print(f'Found files: {len(all_files)}, work: {len(work_list)}')

    threads = []
    lock = threading.Lock()
    finished = [0, 0]
    for thread_index in range(threads_count):
        t = threading.Thread(target=one_thread, args=(work_list, thread_index, dest_dir, lock, finished, use_cuda))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()
    print(f'Finish working. Errors: {finished[1]}')

def log(lock, message):
    lock.acquire()
    print(message)
    lock.release()

def one_thread(work_list, thread_index, dest_dir, lock, finished, use_cuda):
    log(lock, f'Start thread {thread_index}')
    tmp_name = os.path.join(dest_dir, f'tmp_convert_{thread_index}.mp4')
    log_name = os.path.join(dest_dir, f'tmp_convert_{thread_index}.log')
    while True:
        lock.acquire()
        index = finished[0]
        finished[0] = index + 1
        lock.release()
        if index >= len(work_list):
            break

        src_name, dst_name = work_list[index]
        size = os.path.getsize(src_name)
        log(lock, f'Thread {thread_index} start {index+1}/{len(work_list)} {src_name}, size = {size // 1024**2} MB')
        time_start = time.time()
        result = convert_one_file(lock, src_name, dst_name, tmp_name, log_name, use_cuda)
        if result is None:
            time_finish = time.time()
            time_work = int(time_finish - time_start)
            str_time = f'{time_work // 3600}:{(time_work // 60) % 60:02d}:{time_work % 60:02d}'
            size_new = os.path.getsize(dst_name)
            log(lock, f'Finish {src_name}, time = {str_time}, size = {size // 1024**2} MB -> {size_new // 1024**2} MB')
        else:
            lock.acquire()
            finished[1] += 1
            lock.release()
    
    log(lock, f'Finish thread {thread_index}')

def convert_one_file(lock, src_name, dst_name, tmp_name, log_name, use_cuda):
    if os.path.exists(tmp_name):
        os.unlink(tmp_name)
    log_file = open(log_name, 'a')
    if use_cuda:
        dop_args1 = '-hwaccel cuda -hwaccel_output_format cuda'
        dop_args2 = '-c:a copy -c:v h264 -preset p2 -tune ll -b:v 0.3M -bufsize 5M -maxrate 10M -qmin 0 -g 250 -bf 3 -b_ref_mode middle -temporal-aq 1 -rc-lookahead 20 -i_qfactor 0.75 -b_qfactor 1.1'
    else:
        dop_args1 = ''
        dop_args2 = '-strict -2 -vcodec libx264 -crf 25'
    cmd_line = f'ffmpeg {dop_args1} -i "{src_name}" {dop_args2} "{tmp_name}"'
    #cmd_line = f'ls -l "{src_name}"'
    result = subprocess.run(cmd_line, stdout=log_file, stderr=log_file, shell=True).returncode
    if result != 0:
        log(lock, f'!!! Error converting file {src_name}, result code: {result}')
        return result
    os.rename(tmp_name, dst_name)
    return None

if __name__ == '__main__':
    if len(sys.argv) == 4:
        _, src_dir, dest_dir, threads_count = sys.argv
        use_cuda = False
    elif len(sys.argv) == 5:
        _, src_dir, dest_dir, threads_count, use_cuda = sys.argv
        if use_cuda != 'cuda':
            how_usage()
            sys.exit(1)
        use_cuda = True
    else:
        how_usage()
        sys.exit(1)

    work(src_dir, dest_dir, threads_count, use_cuda)
