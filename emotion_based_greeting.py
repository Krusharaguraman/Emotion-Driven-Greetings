import cv2
from deepface import DeepFace

# Function to generate unique, lengthy greetings based on the detected emotion
def get_unique_greeting(emotion):
    greetings = {
        "happy": "Hey there! 😊 It seems like you're absolutely glowing with happiness today, and honestly, it's contagious! Your smile is lighting up the room, and I just want to say how amazing it is to see you so positive and upbeat. Keep embracing that joy, because the world needs more people like you who spread good vibes wherever they go. Whether you're accomplishing something great or just enjoying the simple moments, remember to keep enjoying the ride. You’re doing great, and today looks like it’s going to be one of those fantastic days. Keep shining! 🌟",
        
        "sad": "Hey, I can see that you’re not feeling your best right now, and I want to acknowledge that. 😔 Sometimes life throws challenges at us, and it's completely okay to feel down. But even in moments like these, remember that you're not alone. It’s perfectly fine to take things slow, to feel vulnerable, and to give yourself the time and space you need. I hope you know how strong you are, even if it doesn’t feel like it today. Healing takes time, but with each step, you're getting closer to brighter days. Take a deep breath, you’ve got this, and I'm here if you need to talk or just need a little reminder that everything will eventually get better.",
        
        "angry": "I can see the frustration and anger in your expression. 😡 It's completely valid to feel this way, especially when things don’t seem to be going your way. But remember, it's okay to feel upset—it’s a natural part of being human. While anger can sometimes feel overwhelming, it's important to acknowledge it, understand what’s triggering it, and then find ways to release it in a healthy way. Maybe take a moment to breathe, step away, or think about what’s really bothering you. It’s okay to feel angry, but don’t let it take away your peace. You are in control of your emotions, and I believe that you'll find the clarity and calmness you need soon. Take care of yourself, and let things settle when you're ready.",
        
        "surprise": "Whoa, it looks like you’ve just encountered something that took you by surprise! 😲 I totally get that feeling of being caught off guard—it can be exciting, disorienting, or even a little overwhelming all at once. Life has a way of throwing unexpected twists our way, and while they can shake us up, they can also lead to amazing opportunities and discoveries. Embrace the moment, take it all in, and remember that surprises—good or bad—are just part of the journey. You’ve got the ability to adapt and thrive, even when things are unexpected. So, take a moment, breathe, and get ready for whatever adventure comes next!",
        
        "fear": "It looks like you might be feeling a little uneasy or fearful right now. 😟 I just want to say that it's okay to feel scared—fear is a natural response to things that feel uncertain or outside of our control. But here's the thing: fear often comes from the unknown, and it’s important to remember that you’re stronger than the things that scare you. Take a deep breath, trust yourself, and know that you are capable of facing what’s ahead, even if it feels daunting. You’ve faced challenges before, and I believe that you have the courage to face whatever this is, too. Take small steps, and soon, you’ll feel more empowered. You are not alone in this.",
        
        "disgust": "I can see that something has really bothered you, and you're feeling a bit disgusted by it. 😖 It's completely okay to feel that way when things don't align with your values or when something simply doesn’t sit right with you. But I want to encourage you to not let that discomfort control your day. Take a moment to reset, focus on something positive, or even distract yourself with something that makes you feel better. Life’s full of things that challenge us and make us uncomfortable, but it's how we choose to respond that shapes the rest of our day. So, take a deep breath and know that this feeling will pass. You are in control of how you move forward from this.",
        
        "neutral": "Hey there! 😊 You seem to be in a pretty calm state, just going with the flow of the day. Sometimes, neutral is exactly what we need—it’s that peaceful balance where everything feels steady and stable. Even when nothing particular stands out, it’s a good reminder that not every moment needs to be filled with excitement or intensity. In these times, you get the opportunity to reflect and recharge, which is just as valuable as the high-energy moments. Enjoy this calm, take in your surroundings, and know that you’re doing great, even when things feel just... steady.",
    }

    return greetings.get(emotion, "Hello! How’s your day going? 😊")

# Initialize the camera
cap = cv2.VideoCapture(0)  # 0 for the default camera

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

print("Press 'SPACE' to take a snapshot and analyze emotion.")

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame.")
        break

    # Display the resulting frame
    cv2.imshow('Press "SPACE" to capture an image', frame)

    # Wait for user to press space bar to capture the image
    key = cv2.waitKey(1) & 0xFF
    if key == 32:  # 32 is the ASCII value for space bar
        # Save the captured image
        cv2.imwrite("captured_image.jpg", frame)
        print("Image captured!")

        # Analyze the image using DeepFace
        result = DeepFace.analyze("captured_image.jpg", actions=['emotion'])

        # Get the detected emotion
        emotion = result[0]['dominant_emotion']
        print(f"Detected emotion: {emotion}")

        # Get the corresponding unique greeting
        greeting = get_unique_greeting(emotion)

        # Print the greeting
        print(greeting)

        break  # Exit the loop after capturing and analyzing

    # Exit the loop if 'q' is pressed
    elif key == ord('q'):
        print("Exiting...")
        break

# Release the camera and close OpenCV window
cap.release()
cv2.destroyAllWindows()
