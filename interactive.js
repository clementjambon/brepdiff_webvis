const results = [
    [
        "0025",
        [
            "/viser-client/",
            "?playbackPath=/recordings/raw_traj/0025.viser",
            "&initialCameraPosition=0.0,-1.0,1.0",
            "&initialCameraLookAt=0.0,0.0,0.0",
            // "&baseSpeed=0.5",
            // "&darkMode",
        ],
        "./recordings/gen_14_1_16.png",
        ["./recordings/raw_traj/0025.mp4",
            "Generated B-rep"]
    ],
    [
        "0111",
        [
            "/viser-client/",
            "?playbackPath=/recordings/raw_traj/0111.viser",
            "&initialCameraPosition=0.0,-1.0,1.0",
            "&initialCameraLookAt=0.0,0.0,0.0",
        ],
        "./recordings/gen_14_1_16.png",
        ["./recordings/raw_traj/0111.mp4",
            "Generated B-rep"]
    ],
    [
        "0133",
        [
            "/viser-client/",
            "?playbackPath=/recordings/raw_traj/0111.viser",
            "&initialCameraPosition=0.0,-1.0,1.0",
            "&initialCameraLookAt=0.0,0.0,0.0",
        ],
        "./recordings/gen_14_1_16.png",
        ["./recordings/raw_traj/0133.mp4",
            "Interpolation"]
    ],
    [
        "interp_teaser",
        [
            "/viser-client/",
            "?playbackPath=/recordings/interp/teaser.viser",
            "&initialCameraPosition=0.0,-1.0,1.0",
            "&initialCameraLookAt=0.0,0.0,0.0",
        ],
        "./recordings/gen_14_1_16.png",
        ["./recordings/interp/0001.png",
            "Interpolation"]
    ],
    [
        "autocompletion_teaser",
        [
            "/viser-client/",
            "?playbackPath=/recordings/autocompletion/teaser.viser",
            "&initialCameraPosition=0.0,-1.0,1.0",
            "&initialCameraLookAt=0.0,0.0,0.0",
        ],
        "./recordings/gen_14_1_16.png",
        ["./recordings/raw_traj/0133.mp4",
            "Completed B-rep"]
    ],
];

function initializeResultSelector(resultsElement) {
    const selectorElement = resultsElement.querySelector(".results-selector");
    const resultsThumbnails = selectorElement.querySelector(
        ".results-thumbnails",
    );
    const prevButton = selectorElement.querySelector(".results-prev");
    const nextButton = selectorElement.querySelector(".results-next");
    let currentIndex = 0;

    function getExtension(filename) {
        const idx = filename.lastIndexOf('.');
        return idx >= 0 ? filename.slice(idx + 1) : '';
    }

    // ===================
    // IFRAME
    // ===================

    function createIframe(src) {
        const iframe = document.createElement("iframe");
        console.log("Creating iframe with src", src);
        iframe.src = src;
        return iframe;
    }

    function showIframe(src) {
        const wrapper = resultsElement.querySelector(".iframe-wrapper");
        wrapper.innerHTML = "";
        const iframe = createIframe(Array.isArray(src) ? src.join("") : src);
        wrapper.appendChild(iframe);
    }

    function hideIframe() {
        const wrapper = resultsElement.querySelector(".iframe-wrapper");
        wrapper.innerHTML = ""; // Remove iframe from DOM
    }

    // ===================
    // VIDEO
    // ===================

    function createVideo(src) {
        const video = document.createElement("video");
        console.log("Creating video with src", src);
        video.src = src;
        // video.controls = false; // No control
        // video.width = 250;
        video.autoplay = true;
        video.loop = true;
        video.playsInline = true;
        return video;
    }

    function createImage(src) {
        const image = document.createElement("img");
        console.log("Creating img with src", src);
        image.src = src;
        return image;
    }

    function showVideo(video_payload) {
        src = video_payload[0];
        caption = video_payload[1];
        if (getExtension(src) == "mp4") {
            const wrapper = resultsElement.querySelector(".video-wrapper");
            wrapper.innerHTML = "";
            const video = createVideo(Array.isArray(src) ? src.join("") : src);
            wrapper.appendChild(video);
        } else {
            const wrapper = resultsElement.querySelector(".video-wrapper");
            wrapper.innerHTML = "";
            const image = createImage(Array.isArray(src) ? src.join("") : src);
            wrapper.appendChild(image);
        }
        const video_caption = resultsElement.querySelector(".video-caption");
        video_caption.innerHTML = caption;
    }

    function hideVideo() {
        const wrapper = resultsElement.querySelector(".video-wrapper");
        wrapper.innerHTML = ""; // Remove iframe from DOM
    }

    function updateSelection(index) {
        if (currentIndex !== index) {
            hideIframe(); // Hide previous iframe
            hideVideo(); // Hide previous video
        }
        currentIndex = index;
        resultsThumbnails
            .querySelectorAll("a")
            .forEach((a, i) =>
                a.setAttribute("data-selected", i === index ? "true" : "false"),
            );

        const selectedThumbnail = resultsThumbnails.children[index];

        // Scroll the selected thumbnail into view
        const thumbnailsContainer = resultsThumbnails;
        const scrollLeft =
            selectedThumbnail.offsetLeft -
            (thumbnailsContainer.clientWidth - selectedThumbnail.clientWidth) / 2;
        thumbnailsContainer.scrollTo({
            left: scrollLeft,
            behavior: "smooth",
        });

        // Update URL with the selected result
        const resultName = results[index][0].toLowerCase().replace(/\s+/g, "-");
        const currentPath = window.location.pathname;
        const newUrl = `${currentPath}?result=${resultName}`;
        history.pushState(null, "", newUrl);

        showIframe(results[index][1]);
        if (results[index].length > 3) {
            showVideo(results[index][3])
        }
    }

    results.forEach(([label, src, thumbnail], index) => {
        const link = document.createElement("a");
        link.href = "#";
        link.setAttribute("data-selected", index === 0 ? "true" : "false");
        link.addEventListener("click", (e) => {
            e.preventDefault();
            updateSelection(index);
        });

        const img = document.createElement("img");
        img.src = thumbnail;
        img.alt =
            "Thumbnail that can be clicked to show a result of our method: " + label;
        img.title = label;

        link.appendChild(img);
        resultsThumbnails.appendChild(link);
    });

    prevButton.addEventListener("click", () => {
        updateSelection((currentIndex - 1 + results.length) % results.length);
    });

    nextButton.addEventListener("click", () => {
        updateSelection((currentIndex + 1) % results.length);
    });

    // Check URL for initial result selection
    const urlParams = new URLSearchParams(window.location.search);
    const initialResult = urlParams.get("result");
    console.log(results[0].length);
    if (initialResult) {
        const index = results.findIndex(
            (result) =>
                result[0].toLowerCase().replace(/\s+/g, "-") === initialResult,
        );
        if (index !== -1) {
            updateSelection(index);
        } else {
            showIframe(results[0][1]);
            if (results[0].length > 3) {
                showVideo(results[0][3])
            }
        }
    } else {
        showIframe(results[0][1]);
        if (results[0].length > 3) {
            showVideo(results[0][3])
        }
    }
}

// Initialize all result on the page
document.querySelectorAll(".results").forEach(initializeResultSelector);