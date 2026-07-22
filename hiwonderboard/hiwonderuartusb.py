import serial
import time

# --- Configuration ---
SERIAL_PORT = '/dev/ttyUSB0'
BAUD_RATE = 9600

def move_single_servo(ser, servo_id, position, move_time_ms=1000):
    """
    Sends a binary frame to move a single Hiwonder HTS-35H servo.
    """
    # Clamp position to valid hardware bounds (0 to 1000)
    position = max(0, min(1000, position))
    
    data_length = 8 
    CMD_MOVE = 0x03
    num_servos = 1
    
    # Build the byte array
    packet = [
        0x55, 0x55,                  # Header
        data_length,                 # Length
        CMD_MOVE,                    # Command
        num_servos,                  # Number of servos
        move_time_ms & 0xFF,         # Time Lower Byte
        (move_time_ms >> 8) & 0xFF,  # Time Higher Byte
        servo_id,                    # Servo ID
        position & 0xFF,             # Position Lower Byte
        (position >> 8) & 0xFF       # Position Higher Byte
    ]
    
    ser.write(bytearray(packet))

# --- Main Interactive Loop ---
if __name__ == "__main__":
    try:
        with serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1) as ser:
            time.sleep(1) # Give the adapter a moment to wake up
            
            print("=== Hiwonder Interactive Control ===")
            print("Enter commands in format: <ID> <Position>")
            print("Example: 1 500")
            print("Type 'q' to quit.")
            print("====================================")
            
            while True:
                # Prompt the user for input
                user_input = input("\nCommand (ID POS): ").strip()
                
                # Check for exit command
                if user_input.lower() == 'q':
                    print("Exiting terminal...")
                    break
                
                try:
                    # Split the input string by spaces
                    parts = user_input.split()
                    
                    if len(parts) != 2:
                        print("Error: Please provide exactly two numbers separated by a space.")
                        continue
                        
                    # Parse the numbers
                    s_id = int(parts[0])
                    pos = int(parts[1])
                    
                    print(f"-> Sending Servo {s_id} to position {pos}")
                    
                    # You can also change the 1000ms move time here if you want it faster/slower
                    move_single_servo(ser, servo_id=s_id, position=pos, move_time_ms=1000)
                    
                except ValueError:
                    print("Error: Invalid input. Please enter integers only.")
                    
    except serial.SerialException as e:
        print(f"Serial port error: {e}. Is another program using the port?")
