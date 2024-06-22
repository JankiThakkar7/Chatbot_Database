const chatInput = document.querySelector(".chat_input textarea");
const sendChatBtn = document.querySelector(".chat_input span#send_btn");
const chatbox = document.querySelector(".chatbox");
const chatbotToggler = document.querySelector(".chatbot_toggler");
const chatbotCloseBtn = document.querySelector(".close_btn");
const radioButtons = document.querySelectorAll(
  ".chat-options input[type='radio']"
);
const chatOptionsDiv = document.querySelector(".chat-options");

let userMessage;
const inputInitHeight = chatInput.scrollHeight;
let optionSelected = false;

const closingResponses = [
  "See you later, thanks for visiting",
  "Have a nice day",
  "Bye! Come back again soon.",
];

const createChatLi = (message, className) => {
  const chatLi = document.createElement("li");
  chatLi.classList.add("chat", className);
  let chatContent =
    className === "outgoing" ? `<p></p>` : `<span class="icon"></span><p></p>`;
  chatLi.innerHTML = chatContent;
  chatLi.querySelector("p").textContent = message;
  return chatLi;
};

const generateResponse = (message) => {
  const API_URL = `${SCRIPT_ROOT}/predict`;
  const requestOptions = {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ message }),
  };

  return fetch(API_URL, requestOptions)
    .then((response) => response.json())
    .then((data) => {
      const isClosingResponse = closingResponses.includes(data.answer);
      return { response: data.answer, isClosingResponse };
    })
    .catch((error) => {
      console.error("Error:", error);
      return {
        response: "Sorry, something went wrong. Please try again later.",
        isClosingResponse: false,
      };
    });
};

const typeMessage = (message, chatLi) => {
  const messageElem = chatLi.querySelector("p");
  let index = 0;
  const typingSpeed = 50;
  const intervalId = setInterval(() => {
    if (index < message.length) {
      messageElem.textContent += message.charAt(index);
      index++;
      chatbox.scrollTo(0, chatbox.scrollHeight);
    } else {
      clearInterval(intervalId);
    }
  }, typingSpeed);
};

const handleChat = () => {
  userMessage = chatInput.value.trim();
  if (!userMessage) return;
  chatInput.value = "";
  chatInput.style.height = `${inputInitHeight}px`;

  const outgoingLi = createChatLi(userMessage, "outgoing");
  chatbox.appendChild(outgoingLi);
  chatbox.scrollTo(0, chatbox.scrollHeight);

  chatOptionsDiv.style.display = "none"; // Hide options div after sending a message

  setTimeout(() => {
    const thinkingLi = createChatLi("Thinking...", "incoming");
    chatbox.appendChild(thinkingLi);
    chatbox.scrollTo(0, chatbox.scrollHeight);

    generateResponse(userMessage).then(({ response, isClosingResponse }) => {
      chatbox.removeChild(thinkingLi);
      const incomingLi = createChatLi("", "incoming");
      chatbox.appendChild(incomingLi);
      typeMessage(response, incomingLi);
      chatbox.scrollTo(0, chatbox.scrollHeight);

      if (isClosingResponse) {
        setTimeout(() => {
          chatOptionsDiv.style.display = "block"; // Show options div after closing response
          const chatOptionsContainer = document.createElement("div");
          chatOptionsContainer.classList.add("chat", "incoming");
          chatOptionsContainer.appendChild(chatOptionsDiv);
          chatbox.appendChild(chatOptionsContainer);
          chatbox.scrollTo(0, chatbox.scrollHeight); // Ensure scroll is at the bottom
        }, 2000);
      }
    });
  }, 600);
};

const isOptionSelected = () => {
  for (let radioButton of radioButtons) {
    if (radioButton.checked) {
      return true;
    }
  }
  return false;
};

chatInput.addEventListener("input", () => {
  chatInput.style.height = `${inputInitHeight}px`;
  chatInput.style.height = `${chatInput.scrollHeight}px`;
});

chatInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey && window.innerWidth > 700) {
    e.preventDefault();
    if (isOptionSelected()) {
      handleChat();
    }
  }
});

sendChatBtn.addEventListener("click", () => {
  if (isOptionSelected()) {
    handleChat();
  }
});

chatbotCloseBtn.addEventListener("click", () =>
  document.body.classList.remove("show_chatbot")
);
chatbotToggler.addEventListener("click", () =>
  document.body.classList.toggle("show_chatbot")
);

radioButtons.forEach((radioButton) => {
  radioButton.addEventListener("change", () => {
    optionSelected = true;
  });
});
