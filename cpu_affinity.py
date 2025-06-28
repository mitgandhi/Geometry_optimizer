import psutil
import platform
import time


def get_available_cores():
    total_cores = psutil.cpu_count()
    if total_cores <= 1:
        raise RuntimeError("At least 2 CPU cores required (core 0 is reserved)")

    available_cores = list(range(1, total_cores))

    print(f"CPU Core Analysis:")
    print(f"  Total cores detected: {total_cores}")
    print(f"  Core 0: RESERVED for system")
    print(f"  Available cores: {available_cores}")
    print(f"  Number of cores for optimization: {len(available_cores)}")

    return available_cores


def set_process_affinity(process_or_pid, core_id):
    if core_id == 0:
        core_id = 1
        print(f"WARNING: Core 0 reassigned to core 1")

    try:
        if platform.system() == "Windows":
            try:
                import win32process
                import win32api

                pid = process_or_pid.pid if hasattr(process_or_pid, 'pid') else process_or_pid
                cpu_mask = 1 << core_id
                print(f"Setting Windows affinity mask to {cpu_mask} (binary: {bin(cpu_mask)}) for core {core_id}")

                process_handle = win32api.OpenProcess(
                    win32process.PROCESS_SET_INFORMATION | win32process.PROCESS_QUERY_INFORMATION,
                    False, pid
                )
                win32process.SetProcessAffinityMask(process_handle, cpu_mask)
                actual_mask, _ = win32process.GetProcessAffinityMask(process_handle)
                win32api.CloseHandle(process_handle)

                if actual_mask & 1:
                    corrected_mask = actual_mask & ~1
                    if corrected_mask == 0:
                        corrected_mask = 1 << core_id

                    process_handle = win32api.OpenProcess(
                        win32process.PROCESS_SET_INFORMATION | win32process.PROCESS_QUERY_INFORMATION,
                        False, pid
                    )
                    win32process.SetProcessAffinityMask(process_handle, corrected_mask)
                    actual_mask, _ = win32process.GetProcessAffinityMask(process_handle)
                    win32api.CloseHandle(process_handle)

                used_cores = [i for i in range(32) if actual_mask & (1 << i)]
                print(f"✓ Windows affinity set to cores {used_cores} (mask: {actual_mask})")
                return 0 not in used_cores

            except ImportError:
                print("win32process not available, falling back to psutil")
            except Exception as e:
                print(f"Windows affinity setting failed: {e}")

        print(f"Using psutil to set affinity to core {core_id}")
        p = psutil.Process(process_or_pid.pid if hasattr(process_or_pid, 'pid') else process_or_pid)
        p.cpu_affinity([core_id])
        actual_affinity = p.cpu_affinity()

        if 0 in actual_affinity:
            corrected_affinity = [c for c in actual_affinity if c != 0] or [core_id]
            p.cpu_affinity(corrected_affinity)
            actual_affinity = p.cpu_affinity()

        print(f"✓ psutil affinity successfully set to cores {actual_affinity}")
        return 0 not in actual_affinity

    except Exception as e:
        print(f"Failed to set CPU affinity: {e}")
        return False


def force_exclude_core0():
    try:
        current_process = psutil.Process()
        current_affinity = current_process.cpu_affinity()

        if 0 in current_affinity:
            new_affinity = [c for c in current_affinity if c != 0] or [get_available_cores()[0]]
            current_process.cpu_affinity(new_affinity)
            print(f"Main process affinity corrected to: {current_process.cpu_affinity()}")
        else:
            print(f"Main process already excludes core 0: {current_affinity}")

    except Exception as e:
        print(f"Warning: Could not set main process affinity: {e}")


def set_multicore_affinity_windows(process_handle, process_id):
    try:
        import win32process
        import win32api

        total_cores = psutil.cpu_count()
        all_cores_except_0 = ((1 << total_cores) - 1) & ~1

        print(f"Setting process (PID {process_id}) to use cores 1-{total_cores - 1}")
        print(f"Affinity mask: {all_cores_except_0} (binary: {bin(all_cores_except_0)})")

        for attempt in range(5):
            try:
                win32process.SetProcessAffinityMask(process_handle, all_cores_except_0)
                actual_mask, _ = win32process.GetProcessAffinityMask(process_handle)

                if actual_mask & 1:
                    corrected_mask = actual_mask & ~1
                    win32process.SetProcessAffinityMask(process_handle, corrected_mask)
                    actual_mask, _ = win32process.GetProcessAffinityMask(process_handle)

                used_cores = [i for i in range(32) if actual_mask & (1 << i)]
                print(f"Attempt {attempt + 1}: Process affinity = {used_cores}")

                if not (actual_mask & 1) and len(used_cores) > 1:
                    print(f"✓ Successfully set process to cores {used_cores} (Core 0 excluded)")
                    return True

                time.sleep(0.1)

            except Exception as e:
                print(f"Affinity attempt {attempt + 1} failed: {e}")
                if attempt == 4:
                    raise RuntimeError("Could not set multi-core affinity for process")

        return False

    except ImportError:
        print("win32process not available")
        return False


def set_multicore_affinity_psutil(process_pid):
    try:
        total_cores = psutil.cpu_count()
        all_cores_except_0 = list(range(1, total_cores))

        for attempt in range(10):
            try:
                time.sleep(0.05)
                p = psutil.Process(process_pid)
                p.cpu_affinity(all_cores_except_0)

                actual_affinity = p.cpu_affinity()
                if 0 not in actual_affinity and len(actual_affinity) > 1:
                    print(f"✓ Successfully set process to cores {actual_affinity} (attempt {attempt + 1})")
                    return True

            except psutil.NoSuchProcess:
                print(f"Process ended during affinity setting")
                return False
            except Exception as e:
                print(f"Affinity attempt {attempt + 1} failed: {e}")

        return False

    except Exception as e:
        print(f"Error setting psutil affinity: {e}")
        return False


def monitor_process_affinity(process_handle, process_pid, check_interval=2.0):
    """Monitor process affinity without timeout - let simulation run until completion"""
    return _monitor_windows_affinity(process_handle, process_pid, check_interval) if platform.system() == "Windows" else _monitor_psutil_affinity(process_pid, check_interval)


def _monitor_windows_affinity(process_handle, process_pid, check_interval=2.0):
    """Monitor Windows process without timeout"""
    import win32process, win32event, win32api

    total_cores = psutil.cpu_count()
    allowed_mask = ((1 << total_cores) - 1) & ~1

    print(f"Monitoring process {process_pid} without timeout - will run until completion")

    try:
        while True:
            # Check if process is still running (non-blocking)
            wait_result = win32event.WaitForSingleObject(process_handle, 0)
            if wait_result == win32event.WAIT_OBJECT_0:
                # Process has finished normally
                print(f"✓ Process {process_pid} completed normally")
                return True
            elif wait_result != win32event.WAIT_TIMEOUT:
                # Some other error
                print(f"⚠️ Process {process_pid} ended with wait result: {wait_result}")
                return False

            # Check and correct affinity periodically
            try:
                current_mask, _ = win32process.GetProcessAffinityMask(process_handle)
                if current_mask & 1:  # Core 0 detected
                    print(f"[Monitor] Correcting core 0 usage for process {process_pid}")
                    win32process.SetProcessAffinityMask(process_handle, allowed_mask)
                    corrected_mask, _ = win32process.GetProcessAffinityMask(process_handle)
                    if corrected_mask & 1:
                        print(f"❌ Failed to exclude core 0 from process {process_pid}")
                        return False
                    else:
                        print(f"✓ Core 0 successfully removed from process {process_pid}")

            except Exception as e:
                # Handle might be invalid - check if process still exists
                try:
                    exit_code = win32process.GetExitCodeProcess(process_handle)
                    if exit_code != win32process.STILL_ACTIVE:
                        print(f"✓ Process {process_pid} finished with exit code: {exit_code}")
                        return exit_code == 0
                except:
                    print(f"❌ Process {process_pid} handle became invalid: {e}")
                    return False

            time.sleep(check_interval)

    except Exception as e:
        print(f"❌ Monitoring error for process {process_pid}: {e}")
        return False


def _monitor_psutil_affinity(process_pid, check_interval):
    """Monitor psutil process without timeout"""
    try:
        total_cores = psutil.cpu_count()
        all_cores_except_0 = list(range(1, total_cores))

        print(f"Monitoring process {process_pid} without timeout - will run until completion")

        while True:
            try:
                p = psutil.Process(process_pid)
                if not p.is_running():
                    print(f"✓ Process {process_pid} completed")
                    break

                # Check affinity periodically
                current_affinity = p.cpu_affinity()
                if 0 in current_affinity:
                    corrected = [c for c in current_affinity if c != 0] or all_cores_except_0
                    p.cpu_affinity(corrected)
                    print(f"[Monitor] Corrected affinity to exclude core 0 for process {process_pid}")

            except (psutil.NoSuchProcess, psutil.AccessDenied):
                print(f"✓ Process {process_pid} ended")
                break
            except Exception as e:
                print(f"⚠️ Monitoring warning for process {process_pid}: {e}")

            time.sleep(check_interval)

        return True

    except Exception as e:
        print(f"❌ Error monitoring process {process_pid}: {e}")
        return False