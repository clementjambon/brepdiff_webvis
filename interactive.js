const results = [
    [
        "gen_14_1_16",
        [
            "https://clementjambon.github.io/brepdiff_webvis/viser-client/",
            "?playbackPath=/recordings/gen_14_1_16.viser",
            // "&synchronizedVideoOverlay=/recordings/006545_mpii_test.mp4",
            // "&synchronizedVideoTimeOffset=0.0",
            "&initialCameraPosition=0.0,-1.0,1.0",
            "&initialCameraLookAt=0.0,0.0,0.0",
            // "&baseSpeed=0.5",
            // "&darkMode",
        ],
        "./recordings/gen_14_1_16.png",
    ],
    [
        "gen_14_1_17",
        [
            "https://clementjambon.github.io/brepdiff_webvis/viser-client",
            "?playbackPath=/recordings/gen_14_1_17.viser",
            // "&synchronizedVideoOverlay=/recordings/006545_mpii_test.mp4",
            // "&synchronizedVideoTimeOffset=0.0",
            "&initialCameraPosition=0.0,-1.0,1.0",
            "&initialCameraLookAt=0.0,0.0,0.0",
            // "&baseSpeed=0.5",
            // "&darkMode",
        ],
        "./recordings/gen_14_1_17.png",
    ],
    [
        "gen_14_1_20",
        [
            "https://clementjambon.github.io/brepdiff_webvis/viser-client/",
            "?playbackPath=/recordings/gen_14_1_20.viser",
            // "&synchronizedVideoOverlay=/recordings/006545_mpii_test.mp4",
            // "&synchronizedVideoTimeOffset=0.0",
            "&initialCameraPosition=0.0,-1.0,1.0",
            "&initialCameraLookAt=0.0,0.0,0.0",
            // "&baseSpeed=0.5",
            // "&darkMode",
        ],
        "./recordings/gen_14_1_20.png",
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

    function updateSelection(index) {
        if (currentIndex !== index) {
            hideIframe(); // Hide previous iframe
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
    if (initialResult) {
        const index = results.findIndex(
            (result) =>
                result[0].toLowerCase().replace(/\s+/g, "-") === initialResult,
        );
        if (index !== -1) {
            updateSelection(index);
        } else {
            showIframe(results[0][1]);
        }
    } else {
        showIframe(results[0][1]);
    }
}

// Initialize all result on the page
document.querySelectorAll(".results").forEach(initializeResultSelector);