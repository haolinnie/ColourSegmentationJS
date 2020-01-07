let utils = new Utils('errorMessage');

let width = 0;
let height = 0;

let resolution = window.innerWidth < 960 ? 'qvga' : 'vga';

// whether streaming video from the camera.
let streaming = false;

let video = document.getElementById('videoInput');
let vc = null;

let container = document.getElementById('container');

let lastFilter = '';
let src = null;
let dstC1 = null;
let dstC3 = null;
let dstC4 = null;

function startVideoProcessing() {
    src = new cv.Mat(height, width, cv.CV_8UC4);
    dstC1 = new cv.Mat(height, width, cv.CV_8UC1);
    dstC3 = new cv.Mat(height, width, cv.CV_8UC3);
    dstC4 = new cv.Mat(height, width, cv.CV_8UC4);
    requestAnimationFrame(processVideo);
}

function passThrough(src) {
    return src;
}

function gray(src) {
    cv.cvtColor(src, dstC1, cv.COLOR_RGBA2GRAY);
    return dstC1;
}

function hsv(src) {
    cv.cvtColor(src, dstC3, cv.COLOR_RGBA2RGB);
    cv.cvtColor(dstC3, dstC3, cv.COLOR_RGB2HSV);
    return dstC3;
}

function canny(src) {
    cv.cvtColor(src, dstC1, cv.COLOR_RGBA2GRAY);
    cv.Canny(dstC1, dstC1, controls.cannyThreshold1, controls.cannyThreshold2,
             controls.cannyApertureSize, controls.cannyL2Gradient);
    return dstC1;
}

function inRange(src) {
    let lowValue = controls.inRangeLow;
    let lowScalar = new cv.Scalar(lowValue, lowValue, lowValue, 255);
    let highValue = controls.inRangeHigh;
    let highScalar = new cv.Scalar(highValue, highValue, highValue, 255);
    let low = new cv.Mat(height, width, src.type(), lowScalar);
    let high = new cv.Mat(height, width, src.type(), highScalar);
    cv.inRange(src, low, high, dstC1);
    low.delete(); high.delete();
    return dstC1;
}

function hsv_masking(src, src4) {
    // Get segmentation thresholds
    let lowScalar = new cv.Scalar(controls.hueMin, controls.satMin, controls.valMin);
    let highScalar = new cv.Scalar(controls.hueMax, controls.satMax, controls.valMax);
    let low = new cv.Mat(height, width, src.type(), lowScalar);
    let high = new cv.Mat(height, width, src.type(), highScalar);
    // Threshold the HSV colour space to produce a mask
    cv.inRange(src, low, high, dstC1);
    low.delete(); high.delete();
    let kernel = cv.Mat.ones(3, 3, cv.CV_8U);
    cv.erode(dstC1, dstC1, kernel);
    kernel.delete();
    cv.threshold(dstC1, dstC1, 60, 255, cv.THRESH_BINARY);
    cv.bitwise_or(src4, src4, dstC4, dstC1);
    return dstC1;
}

function threshold(src) {
    cv.threshold(src, dstC4, controls.thresholdValue, 200, cv.THRESH_BINARY);
    return dstC4;
}

function adaptiveThreshold(src) {
    let mat = new cv.Mat(height, width, cv.CV_8U);
    cv.cvtColor(src, mat, cv.COLOR_RGBA2GRAY);
    cv.adaptiveThreshold(mat, dstC1, 200, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                         cv.THRESH_BINARY, Number(controls.adaptiveBlockSize), 2);
    mat.delete();
    return dstC1;
}

function gaussianBlur(src) {
    cv.GaussianBlur(src, dstC4,
                    {width: controls.gaussianBlurSize, height: controls.gaussianBlurSize},
                    0, 0, cv.BORDER_DEFAULT);
    return dstC4;
}

function bilateralFilter(src) {
    let mat = new cv.Mat(height, width, cv.CV_8UC3);
    cv.cvtColor(src, mat, cv.COLOR_RGBA2RGB);
    cv.bilateralFilter(mat, dstC3, controls.bilateralFilterDiameter, controls.bilateralFilterSigma,
                       controls.bilateralFilterSigma, cv.BORDER_DEFAULT);
    mat.delete();
    return dstC3;
}

function medianBlur(src) {
    cv.medianBlur(src, dstC4, controls.medianBlurSize);
    return dstC4;
}

function sobel(src) {
    let mat = new cv.Mat(height, width, cv.CV_8UC1);
    cv.cvtColor(src, mat, cv.COLOR_RGB2GRAY, 0);
    cv.Sobel(mat, dstC1, cv.CV_8U, 1, 0, controls.sobelSize, 1, 0, cv.BORDER_DEFAULT);
    mat.delete();
    return dstC1;
}

function scharr(src) {
    let mat = new cv.Mat(height, width, cv.CV_8UC1);
    cv.cvtColor(src, mat, cv.COLOR_RGB2GRAY, 0);
    cv.Scharr(mat, dstC1, cv.CV_8U, 1, 0, 1, 0, cv.BORDER_DEFAULT);
    mat.delete();
    return dstC1;
}

function laplacian(src) {
    let mat = new cv.Mat(height, width, cv.CV_8UC1);
    cv.cvtColor(src, mat, cv.COLOR_RGB2GRAY);
    cv.Laplacian(mat, dstC1, cv.CV_8U, controls.laplacianSize, 1, 0, cv.BORDER_DEFAULT);
    mat.delete();
    return dstC1;
}

let contoursColor = [];
for (let i = 0; i < 10000; i++) {
    contoursColor.push([Math.round(Math.random() * 255),
                        Math.round(Math.random() * 255),
                        Math.round(Math.random() * 255), 255]);
}

function contours(src) {
    cv.cvtColor(src, dstC1, cv.COLOR_RGBA2GRAY);
    cv.threshold(dstC1, dstC4, 120, 200, cv.THRESH_BINARY);
    let contours = new cv.MatVector();
    let hierarchy = new cv.Mat();
    cv.findContours(dstC4, contours, hierarchy,
                    Number(controls.contoursMode),
                    Number(controls.contoursMethod), {x: 0, y: 0});
    dstC3.delete();
    dstC3 = cv.Mat.ones(height, width, cv.CV_8UC3);
    for (let i = 0; i<contours.size(); ++i) {
        let color = contoursColor[i];
        cv.drawContours(dstC3, contours, i, color, 1, cv.LINE_8, hierarchy);
    }
    contours.delete(); hierarchy.delete();
    return dstC3;
}

function hsv_detection(hsv_mask, src) {
    let contours = new cv.MatVector();
    let hierarchy = new cv.Mat();
    cv.findContours(hsv_mask, contours, hierarchy,
                    cv.RETR_TREE,
                    cv.CHAIN_APPROX_SIMPLE, {x: 0, y: 0});

    let area = 0;
    let areaMax = height * width / 30;
    let idxMax = 0;
    for (let i=0; i<contours.size(); i++) {
        area = cv.contourArea(contours.get(i));
        if (area > areaMax) {
            areaMax = area;
            idxMax = i;
        }
    }
    area = 0.3 * areaMax;

    if (contours.size()>0){
        for (let i = 0; i<contours.size(); ++i) {
            let color = contoursColor[i];
            if (cv.contourArea(contours.get(i)) > area)
                // Draw contour
                cv.drawContours(src, contours, i, color, 1, cv.LINE_8, hierarchy);
                // Calculate centroid
                let Moments = cv.moments(contours.get(idxMax), false);
                let cent_x = Math.round(Moments.m10 / Moments.m00);
                let cent_y = Math.round(Moments.m01 / Moments.m00);
                // Draw centroid
                cv.circle(src, {x: cent_x, y: cent_y}, 5, [0, 255, 0, 0], -1);
                cv.putText(src, "("+cent_x+", "+cent_y+")", {x: cent_x-25, y: cent_y-25},
                        cv.FONT_HERSHEY_SIMPLEX, 0.5, [255, 255, 255, 255], 2);
                }
    }

    contours.delete(); hierarchy.delete();
    return src;

}

let base;

function erosion(src) {
    let kernelSize = controls.erosionSize;
    let kernel = cv.Mat.ones(kernelSize, kernelSize, cv.CV_8U);
    let color = new cv.Scalar();
    cv.erode(src, dstC4, kernel, {x: -1, y: -1}, 1, Number(controls.erosionBorderType), color);
    kernel.delete();
    return dstC4;
}

function dilation(src) {
    let kernelSize = controls.dilationSize;
    let kernel = cv.Mat.ones(kernelSize, kernelSize, cv.CV_8U);
    let color = new cv.Scalar();
    cv.dilate(src, dstC4, kernel, {x: -1, y: -1}, 1, Number(controls.dilationBorderType), color);
    kernel.delete();
    return dstC4;
}

function morphology(src) {
    let kernelSize = controls.morphologySize;
    let kernel = cv.getStructuringElement(Number(controls.morphologyShape),
                                          {width: kernelSize, height: kernelSize});
    let color = new cv.Scalar();
    let op = Number(controls.morphologyOp);
    let image = src;
    if (op === cv.MORPH_GRADIENT || op === cv.MORPH_TOPHAT || op === cv.MORPH_BLACKHAT) {
        cv.cvtColor(src, dstC3, cv.COLOR_RGBA2RGB);
        image = dstC3;
    }
    cv.morphologyEx(image, dstC4, op, kernel, {x: -1, y: -1}, 1,
                    Number(controls.morphologyBorderType), color);
    kernel.delete();
    return dstC4;
}

function processVideo() {
    if (!streaming) return;
    stats.begin();
    vc.read(src);
    let result;
    switch (controls.filter) {
        case 'passThrough': result = passThrough(src); break;
        case 'gray': result = gray(src); break;
        case 'green': pass;
        case 'blue': pass;
        case 'cyan': pass;
        case 'purple': pass;
        case 'hsvMask': 
            result = hsv(src);
            result = hsv_masking(result, src);
            result = hsv_detection(result, src);
            break;
        default: result = passThrough(src);
    }

    cv.putText(result, 'Current Mode: '+filters[controls.filter], {x:10,y:20},
               cv.FONT_HERSHEY_SIMPLEX, 0.5, [0,255,0,255],2);

    cv.imshow('canvasOutput', result);
    stats.end();
    lastFilter = controls.filter;
    requestAnimationFrame(processVideo);
}
let stats = null;

let filters = {
    'passThrough': 'Pass Through',
    'hsvMask': 'HSV Mask',
    'gray': 'Gray',
    'hsv': 'HSV',
};

let controls;

function initUI() {
    stats = new Stats();
    stats.showPanel(0);
    container.appendChild(stats.domElement);
    stats.domElement.style.position = 'absolute';
    stats.domElement.style.right = '0px';
    stats.domElement.style.top = '0px';
    
    function setThresh(hM, sM, vM, hm, sm, vm) {
        hueMax.setValue(hM);
        satMax.setValue(sM);
        valMax.setValue(vM);
        hueMin.setValue(hm);
        satMin.setValue(sm);
        valMin.setValue(vm);
    }

    controls = {
        draw: true,
        filter: 'passThrough',
        setFilter: function(filter) {
            this.filter = filter;
        },
        passThrough: function() {
            this.setFilter('passThrough');
            this.draw = false;
        },
        reset: function() {
            setThresh(255,255,255,0,0,0);
        },
        cyan: function() {
            this.setFilter('hsvMask');
            setThresh(255, 255, 255, 50, 40, 20);
        },
        blue: function() {
            this.setFilter('hsvMask');
            setThresh(135, 228, 255, 50, 80, 40);
        },
        purple: function() {
            this.setFilter('hsvMask');
            setThresh(160, 255, 255, 118, 62, 40);
        },
        magenta: function() {
            this.setFilter('hsvMask');
            setThresh(255, 255, 255, 140, 110, 100);
        },
        gray: function() {
            this.setFilter('gray');
        },
        hsv: function() {
            this.setFilter('hsv');
        },
        inRange: function() {
            this.setFilter('inRange');
        },
        hsvMask: function() {
            this.setFilter('hsvMask');
            this.draw = true;
        },
        inRangeLow: 75,
        inRangeHigh: 150,
        hueMax: 255,
        satMax: 255,
        valMax: 255,
        hueMin: 0,
        satMin: 0,
        valMin: 0,
        threshold: function() {
            this.setFilter('threshold');
        },
        thresholdValue: 100,
        adaptiveThreshold: function() {
            this.setFilter('adaptiveThreshold');
        },
        adaptiveBlockSize: 3,
        gaussianBlur: function() {
            this.setFilter('gaussianBlur');
        },
        gaussianBlurSize: 7,
        medianBlur: function() {
            this.setFilter('medianBlur');
        },
        medianBlurSize: 5,
        bilateralFilter: function() {
            this.setFilter('bilateralFilter');
        },
        bilateralFilterDiameter: 5,
        bilateralFilterSigma: 75,
        sobel: function() {
            this.setFilter('sobel');
        },
        sobelSize: 3,
        scharr: function() {
            this.setFilter('scharr');
        },
        laplacian: function() {
            this.setFilter('laplacian');
        },
        laplacianSize: 3,
        canny: function() {
            this.setFilter('canny');
        },
        cannyThreshold1: 150,
        cannyThreshold2: 300,
        cannyApertureSize: 3,
        cannyL2Gradient: false,
        contours: function() {
            this.setFilter('contours');
        },
        contoursMode: cv.RETR_CCOMP,
        contoursMethod: cv.CHAIN_APPROX_SIMPLE,
        morphology: function() {
            this.setFilter('morphology');
        },
        morphologyShape: cv.MORPH_RECT,
        morphologyOp: cv.MORPH_ERODE,
        morphologySize: 5,
        morphologyBorderType: cv.BORDER_CONSTANT,
    };

    let gui = new dat.GUI({autoPlace: false});
    let guiContainer = document.getElementById('guiContainer');
    guiContainer.appendChild(gui.domElement);

    let lastFolder = null;
    function closeLastFolder(folder) {
        if (lastFolder != null && lastFolder != folder) {
            lastFolder.close();
        }
        lastFolder = folder;
    }

    gui.add(controls, 'passThrough').name(filters['passThrough']).onChange(function() {
        closeLastFolder(null);
    });
    gui.add(controls, 'reset').name('Reset');

    let inRange = gui.addFolder("HSV Thresholding");
    inRange.domElement.onclick = function() {
        controls.hsvMask();
    }
    inRange.open();

    let hueMax = inRange.add(controls, 'hueMax', 0, 255, 1).name('Hue Max');
    let satMax = inRange.add(controls, 'satMax', 0, 255, 1).name('Saturation Max');
    let valMax = inRange.add(controls, 'valMax', 0, 255, 1).name('Value Max');
    let hueMin = inRange.add(controls, 'hueMin', 0, 255, 1).name('Hue Min');
    let satMin = inRange.add(controls, 'satMin', 0, 255, 1).name('Saturation Min');
    let valMin = inRange.add(controls, 'valMin', 0, 255, 1).name('Value Min');

    let presets = gui.addFolder("Presets");
    presets.open();
    presets.add(controls, 'cyan');
    presets.add(controls, 'blue');
    presets.add(controls, 'purple');
    presets.add(controls, 'magenta');

}

function startCamera() {
    if (!streaming) {
        utils.clearError();
        utils.startCamera(resolution, onVideoStarted, 'videoInput');
    } else {
        utils.stopCamera();
        onVideoStopped();
    }
}

function onVideoStarted() {
    height = video.videoHeight;
    width = video.videoWidth;
    video.setAttribute('width', width);
    video.setAttribute('height', height);
    streaming = true;
    vc = new cv.VideoCapture(video);
    startVideoProcessing();
}

function stopVideoProcessing() {
    if (src != null && !src.isDeleted()) src.delete();
    if (dstC1 != null && !dstC1.isDeleted()) dstC1.delete();
    if (dstC3 != null && !dstC3.isDeleted()) dstC3.delete();
    if (dstC4 != null && !dstC4.isDeleted()) dstC4.delete();
}

function onVideoStopped() {
    if (!streaming) return;
    stopVideoProcessing();
    document.getElementById('canvasOutput').getContext('2d').clearRect(0, 0, width, height);
    streaming = false;
}

function onOpenCVReady() {
    if (cv.getBuildInformation) {
        console.log(cv.getBuildInformation());
    }
    initUI();
    startCamera();
}

