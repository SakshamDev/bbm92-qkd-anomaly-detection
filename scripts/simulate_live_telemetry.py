import socket
import time
import json
import random

UDP_IP = "127.0.0.1"
UDP_PORT = 5555

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

print(f"📡 Starting UDP Test Transmitter to {UDP_IP}:{UDP_PORT}...")
print("Press Ctrl+C to stop.\n")

attack_timer = 0

try:
    while True:
        # Generate DRDO realistic "normal" physics data
        payload = {
            "qber": round(random.uniform(0.090, 0.098), 4),
            "bell_S": round(random.uniform(2.28, 2.30), 3),
            "coincidence_rate": round(random.uniform(205, 215), 1),
            "visibility": round(random.uniform(0.80, 0.82), 3),
            "channel_loss_dB": round(random.uniform(16.5, 17.5), 2),
            "detection_rate": round(random.uniform(610, 630), 1)
        }
        
        # 5% chance to START a hacker attack
        if attack_timer == 0 and random.random() < 0.05:
            print("\n🚨 INJECTING FAKE HACKER ATTACK (15 SECONDS)!")
            attack_timer = 15
            
        # If an attack is active, inject a subtle sub-threshold attack
        if attack_timer > 0:
            payload["qber"] = round(random.uniform(0.105, 0.115), 4) # Subtle QBER shift
            payload["bell_S"] = round(random.uniform(2.20, 2.25), 3) # Subtle entanglement drop
            payload["coincidence_rate"] = round(random.uniform(190, 200), 1)
            attack_timer -= 1
            if attack_timer == 0:
                print("✅ Attack finished, returning to normal.")
            
        json_data = json.dumps(payload).encode('utf-8')
        sock.sendto(json_data, (UDP_IP, UDP_PORT))
        
        print(f"Sent: {payload}")
        time.sleep(1.0) # Send at exactly 1 Hz

except KeyboardInterrupt:
    print("\n🛑 Transmitter stopped.")
finally:
    sock.close()
