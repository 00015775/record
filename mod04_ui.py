import os
from mod02_storage import (
    list_signers, add_sign, add_topic,
    load_sign_list, load_topic_list,
    recorded_signs, count_repetitions,
)

GREEN = "\033[92m"
WHITE = "\033[97m"
RESET = "\033[0m"


# ── Signer selection ──────────────────────────────────────────────────────────

def select_signer() -> str:
    while True:
        os.system('cls' if os.name == 'nt' else 'clear')
        signers = list_signers()
        print("=== SIGNER MENU ===")
        if signers:
            print("Existing signers: " + "  ".join(signers))
        else:
            print("(no signers yet)")
        print("\nEnter a signer ID (e.g. signer01) or 'q' to quit.\n")
        inp = input("Signer ID: ").strip()
        if inp.lower() == 'q':
            exit(0)
        if not inp.startswith("signer") or not inp[6:].isdigit():
            print("Invalid format — must be signerXX where XX is a number.")
            input("Press Enter to continue...")
            continue
        return inp


# ── Topic selection ───────────────────────────────────────────────────────────

def select_topic(signer_id: str) -> str | None:
    while True:
        os.system('cls' if os.name == 'nt' else 'clear')
        topics = load_topic_list()
        print(f"=== TOPIC MENU  (Signer: {signer_id}) ===\n")

        COLS = 3
        ROWS = (len(topics) + COLS - 1) // COLS
        for row in range(ROWS):
            line = []
            for col in range(COLS):
                idx = row + col * ROWS
                if idx < len(topics):
                    line.append(f"{idx+1:3d}. {topics[idx].ljust(32)}")
            print("  ".join(line))

        print("\n[a] Add new topic")
        print("[b] Back to signer menu")
        print("[q] Quit\n")
        choice = input("Select number, a, b or q: ").strip().lower()

        if choice == 'q':
            exit(0)
        if choice == 'b':
            return None
        if choice == 'a':
            new = input("New topic name: ").strip()
            if new:
                add_topic(new)
            continue
        if choice.isdigit() and 1 <= int(choice) <= len(topics):
            return topics[int(choice) - 1]
        print("Invalid selection.")
        input("Press Enter...")


# ── Sign selection ────────────────────────────────────────────────────────────

def select_sign(signer_id: str, topic: str) -> str | None:
    while True:
        os.system('cls' if os.name == 'nt' else 'clear')
        signs = load_sign_list(topic)
        recorded = recorded_signs(topic, signer_id)

        print(f"=== SIGN LIST  (Signer: {signer_id} | Topic: {topic}) ===\n")
        if signs:
            COLS = 5
            ROWS = (len(signs) + COLS - 1) // COLS
            for row in range(ROWS):
                line = []
                for col in range(COLS):
                    idx = row + col * ROWS
                    if idx < len(signs):
                        s = signs[idx]
                        reps = count_repetitions(topic, signer_id, s)
                        color = GREEN if s in recorded else WHITE
                        label = f"{s} ({reps})" if reps else s
                        line.append(f"{idx+1:3d}. {color}{label.ljust(20)}{RESET}")
                    else:
                        line.append(" " * 24)
                print("  ".join(line))
        else:
            print("  (no signs in this topic yet — press 'a' to add one)")

        print("\n[a] Add new word")
        print("[b] Back to topic menu")
        print("[q] Quit\n")
        choice = input("Select number, a, b or q: ").strip().lower()

        if choice == 'q':
            exit(0)
        if choice == 'b':
            return None
        if choice == 'a':
            new = input("New sign word: ").strip()
            if new:
                add_sign(topic, new)
            continue
        if choice.isdigit() and 1 <= int(choice) <= len(signs):
            return signs[int(choice) - 1]
        print("Invalid selection.")
        input("Press Enter...")


# ── Post-recording menu ───────────────────────────────────────────────────────

def after_recording_menu(signer_id: str, topic: str, sign: str, rep_idx: int) -> str:
    while True:
        os.system('cls' if os.name == 'nt' else 'clear')
        print(f"Finished rep {rep_idx + 1} — '{sign}'  (Topic: {topic})")
        print("[s] Record another repetition")
        print("[d] Done — back to sign list")
        key = input("Choice: ").strip().lower()
        if key == 's':
            return "again"
        if key == 'd':
            return "done"
        print("Press s or d.")
