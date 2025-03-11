import customtkinter as ctk
import threading
import time
import tkinter as tk
from langchain_groq import ChatGroq
from langchain.schema import SystemMessage, AIMessage, HumanMessage

# Initialize the model
model = ChatGroq(model="llama-3.3-70b-versatile", groq_api_key="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")

class AIChatbotWidget(ctk.CTk):
    def __init__(self):
        super().__init__()

        # Window Configurations
        self.title("CHOMP")
        self.geometry("300x300")  # Set window size to 300x300
        self.configure(bg="black")
        self.attributes("-topmost", True)
        self.resizable(False, False)  # Fixed size window

        # Chat Display (Allow Copying but No Editing)
        self.chat_display = ctk.CTkTextbox(self, height=150, width=280, wrap="word", corner_radius=10,
                                           fg_color=("black", "gray20"), text_color="white")
        self.chat_display.pack(padx=10, pady=5, fill="both", expand=True)

        # Enable text selection and copying
        self.chat_display.configure(state="normal")  # Allow text selection
        self.chat_display.bind("<Key>", lambda e: "break")  # Prevent user input
        self.chat_display.bind("<Control-c>", self.copy_selected_text)  # Enable Ctrl+C
        self.chat_display.bind("<Button-3>", self.show_context_menu)  # Right-click context menu

        # User Input Field
        self.user_input = ctk.CTkEntry(self, placeholder_text="Type here...", height=30, corner_radius=15,
                                       fg_color=("black", "gray30"))
        self.user_input.pack(padx=10, pady=5, fill="x")
        self.user_input.bind("<Return>", self.process_input)

        # Buttons Frame
        button_frame = ctk.CTkFrame(self, fg_color="transparent")
        button_frame.pack(padx=10, pady=5, fill="x")

        self.send_button = ctk.CTkButton(button_frame, text="Send", command=self.process_input,
                                         corner_radius=10, fg_color="#4CAF50", width=50)
        self.send_button.pack(side="left", expand=True, padx=2)

        self.close_button = ctk.CTkButton(button_frame, text="Close", command=self.quit,
                                          fg_color="#E74C3C", corner_radius=10, width=50)
        self.close_button.pack(side="right", expand=True, padx=2)

        # Context menu (Right-click)
        self.context_menu = tk.Menu(self, tearoff=0)
        self.context_menu.add_command(label="Copy", command=self.copy_selected_text)

        # Conversation History
        self.message_history = [
            SystemMessage(content="You are a helpful assistant named CHOMP.")
        ]

    def process_input(self, event=None):
        """Handles user input and AI responses."""
        user_text = self.user_input.get().strip()
        if user_text:
            self.update_chat("You: " + user_text)
            self.user_input.delete(0, "end")

            threading.Thread(target=self.display_ai_response, args=(user_text,), daemon=True).start()

    def display_ai_response(self, user_text):
        """Fetch AI response while maintaining conversation history."""
        self.message_history.append(HumanMessage(content=user_text))
        response = self.get_ai_response()
        self.message_history.append(AIMessage(content=response))
        self.typing_effect(f"AI: {response}")

    def get_ai_response(self):
        """Fetch AI response while keeping conversation context."""
        try:
            output = model.invoke(self.message_history)
            return output.content
        except Exception as e:
            return f"Error: {e}"

    def typing_effect(self, message):
        """Simulates typing effect when displaying AI response."""
        self.chat_display.configure(state="normal")  # Allow editing temporarily
        self.chat_display.insert("end", "\n")
        for char in message:
            self.chat_display.insert("end", char)
            self.chat_display.yview("end")
            self.update()  # Forces UI update
            time.sleep(0.03)  # Adjust speed (lower = faster)
        self.chat_display.insert("end", "\n\n")
        self.chat_display.configure(state="disabled")  # Disable editing after typing

    def update_chat(self, message):
        """Updates chat display immediately."""
        self.chat_display.configure(state="normal")  # Allow modifications
        self.chat_display.insert("end", message + "\n\n")
        self.chat_display.configure(state="disabled")  # Disable editing
        self.chat_display.yview("end")

    def copy_selected_text(self, event=None):
        """Copies selected text to clipboard."""
        try:
            selected_text = self.chat_display.selection_get()
            self.clipboard_clear()
            self.clipboard_append(selected_text)
            self.update()  # Update clipboard
        except:
            pass  # Ignore if no text is selected

    def show_context_menu(self, event):
        """Displays a right-click menu for copying text."""
        self.context_menu.post(event.x_root, event.y_root)

if __name__ == "__main__":
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("green")

    chatbot = AIChatbotWidget()
    chatbot.mainloop()
