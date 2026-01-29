async function fileToImageData(file) {
    const img = new Image();
    img.src = URL.createObjectURL(file);
    
    await new Promise((resolve) => (img.onload = resolve)); // 等待图片加载完成
    
    const canvas = document.createElement("canvas");
    canvas.width = img.width;
    canvas.height = img.height;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(img, 0, 0);
    
    return ctx.getImageData(0, 0, canvas.width, canvas.height); // 返回 ImageData
}

function imageDataToBlob(imageData, type = "image/png", quality = 1.0) {
    return new Promise((resolve, reject) => {
        const canvas = document.createElement("canvas");
        canvas.width = imageData.width;
        canvas.height = imageData.height;
        const ctx = canvas.getContext("2d");
        ctx.putImageData(imageData, 0, 0); // 将 ImageData 放入画布
        canvas.toBlob(
            (blob) => {
                if (blob) {
                    resolve(blob); // 成功时返回 Blob
                } else {
                    reject(new Error("Failed to convert ImageData to Blob"));
                }
            },
            type,
            quality
        );
    });
}

function addPaddingToImageData(originalData, originalWidth, originalHeight, extraCols, extraRows) {
    const newWidth = originalWidth + extraCols;
    const newHeight = originalHeight + extraRows;
    const newData = new Uint8ClampedArray(newWidth * newHeight * 4);
    
    for (let y = 0; y < newHeight; y++) {
        for (let x = 0; x < newWidth; x++) {
            const newIndex = 4 * (x + y * newWidth);
            if (y < originalHeight && x < originalWidth) {
                // 复制原图数据
                const oldIndex = 4 * (x + y * originalWidth);
                newData.set(originalData.slice(oldIndex, oldIndex + 4), newIndex);
            } else if (y < originalHeight) {
                // 填充右侧列：复制当前行的最后一列
                const lastColIndex = 4 * (originalWidth - 1 + y * originalWidth);
                newData.set(originalData.slice(lastColIndex, lastColIndex + 4), newIndex);
            } else {
                // 填充底部行：复制最后一行的数据
                const lastRowY = originalHeight - 1;
                const sourceX = Math.min(x, originalWidth - 1); // 确保不超出原图宽度
                const lastRowIndex = 4 * (sourceX + lastRowY * originalWidth);
                newData.set(originalData.slice(lastRowIndex, lastRowIndex + 4), newIndex);
            }
        }
    }
    
    return newData;
}

function cropImageData(imageData, removeColumns, removeRows) {
    const originalWidth = imageData.width;
    const originalHeight = imageData.height;
    
    // 新的宽度和高度
    const newWidth = originalWidth - removeColumns;
    const newHeight = originalHeight - removeRows;
    
    // 创建新的 ImageData
    const croppedData = new Uint8ClampedArray(newWidth * newHeight * 4); // 每个像素占 4 个字节 (RGBA)
    const originalData = imageData.data;
    
    // 遍历原始数据并拷贝需要保留的像素到新的数据中
    for (let y = 0; y < newHeight; y++) {
        for (let x = 0; x < newWidth; x++) {
            const newIndex = (y * newWidth + x) * 4; // 新数据索引
            const oldIndex = (y * originalWidth + x) * 4; // 原始数据索引
            
            // 复制 RGBA 值
            croppedData[newIndex] = originalData[oldIndex];       // R
            croppedData[newIndex + 1] = originalData[oldIndex + 1]; // G
            croppedData[newIndex + 2] = originalData[oldIndex + 2]; // B
            croppedData[newIndex + 3] = originalData[oldIndex + 3]; // A
        }
    }
    
    // 创建新的 ImageData 对象
    return new ImageData(croppedData, newWidth, newHeight);
}