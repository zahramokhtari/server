import threading
from tkinter import *
from tkinter import scrolledtext
from werkzeug.serving import make_server
import os
import flask
import xml.etree.ElementTree as ElementTree
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

app = flask.Flask(__name__)

# Load the dataset
file_path = os.path.join(os.path.dirname(__file__), 'pom1.xml')
with open(file_path, 'r', encoding="utf-8", errors="ignore") as f:
    data = f.read().split("\n----------------------------------------------\n")

# Remove any text or whitespace before the XML declaration
for i, pom in enumerate(data):
    match = re.search(r"<\?xml", pom)
    if match is not None:
        data[i] = pom[match.start():]

# Preprocess the data
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data)

# Train the k-nearest neighbors model
knn_model = NearestNeighbors(n_neighbors=2, algorithm='brute', metric='cosine')
knn_model.fit(X)


# Define the endpoint for the recommendation API
@app.route("/recommend", methods=["POST"])
def recommend():
    try:
        # Get the user's pom.xml file from the request
        user_pom = flask.request.json["pom"]

        # Parse the user's pom.xml file and extract the list of dependencies
        root = ElementTree.fromstring(user_pom)
        dependencies = []
        for dependency in root.iter("{http://maven.apache.org/POM/4.0.0}dependency"):
            group_id_elem = dependency.find("{http://maven.apache.org/POM/4.0.0}groupId")
            artifact_id_elem = dependency.find("{http://maven.apache.org/POM/4.0.0}artifactId")
            if group_id_elem is not None and artifact_id_elem is not None:
                coordinates = group_id_elem.text + ":" + artifact_id_elem.text
                dependencies.append(coordinates)

        # Preprocess the user's pom.xml file
        user_text = " ".join(dependencies)
        user_vec = vectorizer.transform([user_text])

        # Find the k-nearest neighbors to the user's pom.xml file
        distances, indices = knn_model.kneighbors(user_vec)

        # Get the indices of the k-nearest neighbors
        indices = indices[0]

        # Get the top 3 most similar pom.xml files
        top_poms = [data[j] for j in indices[:2]]

        # Extract the list of libraries used in the top 3 most similar pom.xml files
        libraries = set()
        for top_pom in top_poms:
            root = ElementTree.fromstring(top_pom)
            for dependency in root.iter("{http://maven.apache.org/POM/4.0.0}dependency"):
                group_id_elem = dependency.find("{http://maven.apache.org/POM/4.0.0}groupId")
                artifact_id_elem = dependency.find("{http://maven.apache.org/POM/4.0.0}artifactId")
                if group_id_elem is not None and artifact_id_elem is not None:
                    coordinates = group_id_elem.text + ":" + artifact_id_elem.text
                    libraries.add(coordinates)

        # Remove the libraries that are already used in the user's pom.xml file
        for dependency in dependencies:
            if dependency in libraries:
                libraries.remove(dependency)

        print(libraries)
        # Return the recommended libraries
        return flask.jsonify({"recommendations": list(libraries)})

    except Exception as e:
        error_message = "Error: " + str(e)
        if "connection refused" in error_message.lower():
            error_message += ".\nMake sure the recommendation server is running."
        return flask.jsonify({"error": error_message})


http_server = make_server('localhost', 5000, app)


def start_server():
    global http_server
    http_server.serve_forever()


def start_server_thread():
    server_thread = threading.Thread(target=start_server)
    server_thread.start()


def start_server_gui():
    start_button["state"] = "disabled"
    stop_button["state"] = "normal"
    output_box.insert(END, "Server started\n")
    start_server_thread()


def stop_server_gui():
    start_button["state"] = "normal"
    stop_button["state"] = "disabled"
    output_box.insert(END, "Server stopped\n")
    global http_server
    http_server.shutdown()


def on_closing():
    if stop_button["state"] == "normal":
        stop_server_gui()
    os._exit(0)
    window.destroy()


if __name__ == "__main__":
    # Create GUI window
    window = Tk()
    window.title("Recommendation Server")

    # Configure columns and rows to resize automatically
    window.grid_columnconfigure(0, weight=1)
    window.grid_rowconfigure(0, weight=1)
    window.grid_rowconfigure(1, weight=1)
    window.grid_rowconfigure(2, weight=1)
    window.grid_rowconfigure(3, weight=1)

    # Create start button
    start_button = Button(window, text="Start Server", command=start_server_gui)
    start_button.grid(column=0, row=2, padx=10, pady=10, sticky=N+S+E+W)

    # Create stop button
    stop_button = Button(window, text="Stop Server", command=stop_server_gui, state="disabled")
    stop_button.grid(column=0, row=3, padx=10, pady=10, sticky=N + S + E + W)

    # Create output box
    output_box = scrolledtext.ScrolledText(window, width=40, height=5, wrap=WORD)
    output_box.grid(column=0, row=0, padx=10, pady=10, sticky=N + S + E + W)

    # Bind the on_closing method to the window close event
    window.protocol("WM_DELETE_WINDOW", on_closing)

    # Start the GUI event loop
    window.mainloop()
