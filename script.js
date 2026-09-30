const questionInput = document.getElementById("question");
const sendButton = document.getElementById("sendButton");
const chat = document.getElementById("chat");

function addMessage(message, type) {

    const messageDiv = document.createElement("div");

    messageDiv.className = "message " + type;

    const content = document.createElement("div");

    content.className = "message-content";

    content.textContent = message;

    messageDiv.appendChild(content);

    chat.appendChild(messageDiv);

    chat.scrollTop = chat.scrollHeight;
}


async function sendQuestion() {

    const question = questionInput.value.trim();

    if (!question) {
        return;
    }

    addMessage(question, "user");

    questionInput.value = "";

    const typing = document.createElement("div");

    typing.className = "message bot";

    typing.id = "typing";

    typing.innerHTML =
        '<div class="message-content typing">Thinking...</div>';

    chat.appendChild(typing);

    chat.scrollTop = chat.scrollHeight;

    try {

        const response = await fetch("/ask", {

            method: "POST",

            headers: {
                "Content-Type": "application/json"
            },

            body: JSON.stringify({
                question: question
            })

        });

        const data = await response.json();

        document.getElementById("typing").remove();

        addMessage(data.answer, "bot");

    } catch (error) {

        document.getElementById("typing").remove();

        addMessage(
            "Sorry, something went wrong. Make sure the server is running.",
            "bot"
        );
    }
}


function askSuggestion(question) {

    questionInput.value = question;

    sendQuestion();

}


sendButton.addEventListener(
    "click",
    sendQuestion
);


questionInput.addEventListener(
    "keydown",
    function(event) {

        if (event.key === "Enter") {

            sendQuestion();

        }

    }
);