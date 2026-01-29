function getPassWD(){
    var step = 1
    var v = 0
    var h = 0
    if (password.value) {
        step = Math.max(1, parseInt(password.value.slice(0, 2)));
        v = parseInt(password.value.slice(2, 3));
        h = parseInt(password.value.slice(3, 4));
    }
    return [step, v, h]
}

function encrypt(img, text, callback) {
    const [step, v, h] = getPassWD()
    const cvs = document.createElement("canvas");
    const width = (cvs.width = img.width);
    const height = (cvs.height = img.height);
    const ctx = cvs.getContext("2d");
    ctx.drawImage(img, 0, 0);
    
    // 初始化原始数据
    let imgData = ctx.getImageData(0, 0, width, height);
    let newImageData = encryptImageData(imgData)
    
    // 将最终结果绘制回画布
    cvs.width += v
    cvs.height += h
    ctx.putImageData(newImageData, 0, 0);
    cvs.toBlob(b => {
        writeTextChunksToNewBlob(b, text, false).then((newFile) => {
            setsrc(URL.createObjectURL(newFile))
            if (callback && typeof callback === 'function') {
                callback(newFile);
            }
        }).catch((error) => {
            console.error("写入新文件时发生错误：", error);
            if (callback) callback(null);
        });
    }, "image/png");
    dl.style.display = "inline-block"
}

function decrypt(img, text, callback) {
    const [step, v, h] = getPassWD()
    const cvs = document.createElement("canvas");
    const width = (cvs.width = img.width - v);
    const height = (cvs.height = img.height - h);
    const ctx = cvs.getContext("2d");
    ctx.drawImage(img, 0, 0);
    
    // 初始化图像数据
    let imgData = ctx.getImageData(0, 0, width, height);
    let newImageData = decryptImageData(imgData)
    
    // 将最终结果写回画布
    ctx.putImageData(newImageData, 0, 0);
    cvs.toBlob(b => {
        writeTextChunksToNewBlob(b, text, true).then((newFile) => {
            setsrc(URL.createObjectURL(newFile))
            if (callback && typeof callback === 'function') {
                callback(newFile);
            }
        }).catch((error) => {
            console.error("写入新文件时发生错误：", error);
            if (callback) callback(null);
        });
    }, "image/png");
}

function encryptImageData(imageData) {
    const [step, v, h] = getPassWD()
    const width = imageData.width;
    const height = imageData.height;
    const data = imageData.data;
    const totalPixels = width * height;
    const curve = gilbert2d(width, height);
    const offset = Math.round((Math.sqrt(5) - 1) / 2 * totalPixels);
    
    // 预计算曲线的原始位置和目标位置索引
    const oldPositions = new Array(totalPixels);
    const newPositions = new Array(totalPixels);
    for (let i = 0; i < totalPixels; i++) {
        const old_pos = curve[i];
        const new_pos = curve[(i + offset) % totalPixels];
        oldPositions[i] = 4 * (old_pos[0] + old_pos[1] * width);
        newPositions[i] = 4 * (new_pos[0] + new_pos[1] * width);
    }
    
    // 混淆数据多次
    const buffer = new Uint8ClampedArray(data.length);
    for (let j = 0; j < step; j++) {
        for (let i = 0; i < totalPixels; i++) {
            const old_p = oldPositions[i];
            const new_p = newPositions[i];
            buffer[new_p] = data[old_p];
            buffer[new_p + 1] = data[old_p + 1];
            buffer[new_p + 2] = data[old_p + 2];
            buffer[new_p + 3] = data[old_p + 3];
        }
        // 替换为混淆后的数据
        data.set(buffer);
    }
    
    let newData = addPaddingToImageData(data, width, height, v, h)
    const newImageData = new ImageData(newData, width + v, height + h);
    
    return newImageData
}

function decryptImageData(imageData) {
    const [step, v, h] = getPassWD()
    const imgData = imageData;
    const width = imgData.width;
    const height = imgData.height;
    const data = imgData.data;
    const totalPixels = width * height;
    
    // 生成曲线和偏移值
    const curve = gilbert2d(width, height);
    const offset = Math.round((Math.sqrt(5) - 1) / 2 * totalPixels);
    
    // 预计算曲线的原始位置和目标位置索引
    const oldPositions = new Array(totalPixels);
    const newPositions = new Array(totalPixels);
    for (let i = 0; i < totalPixels; i++) {
        const old_pos = curve[i];
        const new_pos = curve[(i + offset) % totalPixels];
        oldPositions[i] = 4 * (old_pos[0] + old_pos[1] * width);
        newPositions[i] = 4 * (new_pos[0] + new_pos[1] * width);
    }
    
    // 解混淆数据多次
    const buffer = new Uint8ClampedArray(data.length); // 临时缓冲区
    for (let j = 0; j < step; j++) {
        for (let i = 0; i < totalPixels; i++) {
            const old_p = oldPositions[i];
            const new_p = newPositions[i];
            buffer[old_p] = data[new_p];
            buffer[old_p + 1] = data[new_p + 1];
            buffer[old_p + 2] = data[new_p + 2];
            buffer[old_p + 3] = data[new_p + 3];
        }
        // 替换为解混淆后的数据
        data.set(buffer);
    }
    
    return imgData
}