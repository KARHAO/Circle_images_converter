/**
 * Auto Images for Adobe Illustrator
 * 版本: v1.1.0
 * 
 * 描述: 此脚本用于批量导入图片到Adobe Illustrator并自动排列对齐
 * 
 * 功能:
 * 1. 批量导入多个图片文件
 * 2. 自动排列图片（网格布局）
 * 3. 可自定义间距、行数和列数
 * 4. 支持多种图片格式（jpg, png, tif, psd等）
 */

// 检查是否有文档打开
if (app.documents.length === 0) {
    alert("请先创建或打开一个文档！");
} else {
    // 主函数
    main();
}

/**
 * 主函数 - 程序入口
 */
function main() {
    // 显示用户界面并获取用户设置
    var settings = showDialog();
    
    // 如果用户点击确定，则继续执行
    if (settings) {
        // 导入图片
        var placedItems = importImages(settings.folder, settings.fileTypes, settings.sortType);
        
        // 如果成功导入了图片，则进行排列
        if (placedItems && placedItems.length > 0) {
            // 排列图片
            arrangeImages(placedItems, settings);
            
            // 显示完成信息
            alert("成功导入并排列了 " + placedItems.length + " 个图片！");
        } else {
            alert("没有找到符合条件的图片文件！");
        }
    }
}

/**
 * 显示用户界面对话框
 * @returns {Object} 用户设置
 */
function showDialog() {
    // 创建对话框
    var dialog = new Window("dialog", "批量导入图片并排列");
    dialog.orientation = "column";
    dialog.alignChildren = ["fill", "top"];
    dialog.spacing = 8;
    dialog.margins = 12;
    
    // 添加文件夹选择区域
    var folderGroup = dialog.add("group");
    folderGroup.orientation = "row";
    folderGroup.alignChildren = ["left", "center"];
    folderGroup.spacing = 8;
    
    folderGroup.add("statictext", undefined, "图片文件夹:");
    var folderText = folderGroup.add("edittext", undefined, "");
    folderText.preferredSize.width = 280;
    
    var browseButton = folderGroup.add("button", undefined, "浏览...");
    
    // 添加文件类型选择区域
    var fileTypesPanel = dialog.add("panel", undefined, "文件类型");
    fileTypesPanel.orientation = "row";
    fileTypesPanel.alignChildren = ["left", "center"];
    fileTypesPanel.margins = 8;
    fileTypesPanel.spacing = 10;
    
    var jpgCheck = fileTypesPanel.add("checkbox", undefined, "JPG");
    jpgCheck.value = true;
    
    var pngCheck = fileTypesPanel.add("checkbox", undefined, "PNG");
    pngCheck.value = true;
    
    var tifCheck = fileTypesPanel.add("checkbox", undefined, "TIF/TIFF");
    tifCheck.value = true;
    
    var psdCheck = fileTypesPanel.add("checkbox", undefined, "PSD");
    psdCheck.value = false;
    
    // 添加排列设置区域
    var arrangePanel = dialog.add("panel", undefined, "排列设置");
    arrangePanel.orientation = "column";
    arrangePanel.alignChildren = ["left", "center"];
    arrangePanel.margins = 8;
    arrangePanel.spacing = 8;
    
    // 布局类型选择
    var layoutGroup = arrangePanel.add("group");
    layoutGroup.orientation = "row";
    layoutGroup.alignChildren = ["left", "center"];
    layoutGroup.spacing = 8;
    
    layoutGroup.add("statictext", undefined, "布局类型:");
    var layoutDropdown = layoutGroup.add("dropdownlist", undefined, ["网格布局", "单行布局", "单列布局"]);
    layoutDropdown.selection = 0;
    
    // 行数和列数设置
    var rowColGroup = arrangePanel.add("group");
    rowColGroup.orientation = "row";
    rowColGroup.alignChildren = ["left", "center"];
    rowColGroup.spacing = 8;
    
    rowColGroup.add("statictext", undefined, "行数:");
    var rowsInput = rowColGroup.add("edittext", undefined, "5");
    rowsInput.preferredSize.width = 40;
    
    rowColGroup.add("statictext", undefined, "列数:");
    var colsInput = rowColGroup.add("edittext", undefined, "8");
    colsInput.preferredSize.width = 40;
    
    // 间距设置
    var spacingGroup = arrangePanel.add("group");
    spacingGroup.orientation = "row";
    spacingGroup.alignChildren = ["left", "center"];
    spacingGroup.spacing = 8;
    
    spacingGroup.add("statictext", undefined, "水平间距:");
    var hSpacingInput = spacingGroup.add("edittext", undefined, "20");
    hSpacingInput.preferredSize.width = 40;
    
    spacingGroup.add("statictext", undefined, "垂直间距:");
    var vSpacingInput = spacingGroup.add("edittext", undefined, "20");
    vSpacingInput.preferredSize.width = 40;
    
    // 位置设置
    var positionGroup = arrangePanel.add("group");
    positionGroup.orientation = "row";
    positionGroup.alignChildren = ["left", "center"];
    positionGroup.spacing = 8;
    
    positionGroup.add("statictext", undefined, "X坐标:");
    var xInput = positionGroup.add("edittext", undefined, "0");
    xInput.preferredSize.width = 40;
    
    positionGroup.add("statictext", undefined, "Y坐标:");
    var yInput = positionGroup.add("edittext", undefined, "0");
    yInput.preferredSize.width = 40;
    
    // 排序设置
    var sortGroup = arrangePanel.add("group");
    sortGroup.orientation = "row";
    sortGroup.alignChildren = ["left", "center"];
    sortGroup.spacing = 8;
    
    sortGroup.add("statictext", undefined, "排序方式:");
    var sortDropdown = sortGroup.add("dropdownlist", undefined, ["名称排序", "修改时间排序", "文件类型排序"]);
    sortDropdown.selection = 0;
    
    // 排序方向设置
    var directionGroup = arrangePanel.add("group");
    directionGroup.orientation = "row";
    directionGroup.alignChildren = ["left", "center"];
    directionGroup.spacing = 8;
    
    directionGroup.add("statictext", undefined, "排序方向:");
    var directionDropdown = directionGroup.add("dropdownlist", undefined, ["横向排序", "竖向排序"]);
    directionDropdown.selection = 0;
    directionDropdown.enabled = (layoutDropdown.selection.index === 0);
    
    // 添加按钮组
    var buttonGroup = dialog.add("group");
    buttonGroup.orientation = "row";
    buttonGroup.alignChildren = ["center", "center"];
    buttonGroup.spacing = 10;
    
    var cancelButton = buttonGroup.add("button", undefined, "取消", {name: "cancel"});
    var okButton = buttonGroup.add("button", undefined, "确定", {name: "ok"});
    
    // 浏览按钮点击事件
    browseButton.onClick = function() {
        var folder = Folder.selectDialog("选择包含图片的文件夹");
        if (folder) {
            folderText.text = folder.fsName;
        }
    };
    
    // 布局类型变更事件
    layoutDropdown.onChange = function() {
        var isGrid = (layoutDropdown.selection.index === 0);
        rowsInput.enabled = isGrid;
        colsInput.enabled = isGrid;
        directionDropdown.enabled = isGrid;
    };
    
    // 显示对话框
    if (dialog.show() === 1) {
        // 用户点击了确定按钮，收集设置
        var fileTypes = [];
        if (jpgCheck.value) fileTypes.push("jpg", "jpeg");
        if (pngCheck.value) fileTypes.push("png");
        if (tifCheck.value) fileTypes.push("tif", "tiff");
        if (psdCheck.value) fileTypes.push("psd");
        
        return {
            folder: folderText.text,
            fileTypes: fileTypes,
            layoutType: layoutDropdown.selection.index,
            rows: parseInt(rowsInput.text) || 3,
            cols: parseInt(colsInput.text) || 4,
            hSpacing: parseInt(hSpacingInput.text) || 20,
            vSpacing: parseInt(vSpacingInput.text) || 20,
            positionX: parseFloat(xInput.text) || 0,
            positionY: parseFloat(yInput.text) || 0,
            sortType: sortDropdown.selection.index,
            sortDirection: directionDropdown.selection.index
        };
    }
    
    return null;
}

// 自然排序函数
function naturalSort(a, b) {
    var ax = [], bx = [];
    a.name.replace(/(\d+)|(\D+)/g, function(_, $1, $2) { ax.push([$1 || Infinity, $2 || ""]) });
    b.name.replace(/(\d+)|(\D+)/g, function(_, $1, $2) { bx.push([$1 || Infinity, $2 || ""]) });
    
    while(ax.length && bx.length) {
        var an = ax.shift();
        var bn = bx.shift();
        var nn = (an[0] - bn[0]) || an[1].localeCompare(bn[1]);
        if(nn) return nn;
    }
    return ax.length - bx.length;
}

/**
 * 导入图片文件
 * @param {string} folderPath - 图片文件夹路径
 * @param {Array} fileTypes - 文件类型数组
 * @param {number} sortType - 排序方式
 * @returns {Array} 导入的图片项数组
 */
function importImages(folderPath, fileTypes, sortType) {
    // 类型安全处理
    var validTypes = fileTypes instanceof Array ? fileTypes : [];
    
    // 检查文件夹路径
    if (!folderPath) {
        alert("请选择一个有效的文件夹！");
        return null;
    }
    
    var folder = new Folder(folderPath);
    if (!folder.exists) {
        alert("所选文件夹不存在！");
        return null;
    }
    
    // 获取文件夹中的所有文件
    var files = folder.getFiles();
    
    // 根据选择的排序方式对文件进行排序
    switch (sortType) {
        case 0: // 名称排序
            files.sort(function(a, b) {
                return a.name.localeCompare(b.name);
            });
            break;
        case 1: // 修改时间排序
            files.sort(function(a, b) {
                return b.modified.getTime() - a.modified.getTime();
            });
            break;
        case 2: // 文件类型排序
            files.sort(function(a, b) {
                var aExt = a.name.split('.').pop().toLowerCase();
                var bExt = b.name.split('.').pop().toLowerCase();
                return aExt.localeCompare(bExt);
            });
            break;
    }
    
    // 按文件名自然排序
    files.sort(naturalSort);
    
    // 筛选符合类型的图片文件
    var imageFiles = [];
    
    for (var i = 0; i < files.length; i++) {
        var file = files[i];
        if (file instanceof File) {
            var extension = file.name.split('.').pop().toLowerCase();
            for (var t = 0; t < validTypes.length; t++) {
                if (validTypes[t] === extension) {
                    imageFiles.push(file);
                    break;
                }
            }
        }
    }
    
    if (imageFiles.length === 0) {
        return [];
    }
    
    // 获取当前文档
    var doc = app.activeDocument;
    
    // 创建一个数组存储导入的图片
    var placedItems = [];
    
    // 导入每个图片（保持原始尺寸）
    for (var j = 0; j < imageFiles.length; j++) {
        try {
            var placedItem = doc.placedItems.add();
            placedItem.file = imageFiles[j];
            
            // 保持原始尺寸（禁用自动变换）
            placedItem.resize(100, 100, true, true, false, true, 0);
            placedItem.position = [0,0];
            
            // 禁用链接嵌入
            if (placedItem.hasOwnProperty('embed')) {
                placedItem.embed = false;
            }
            
            placedItems.push(placedItem);
        } catch (e) {
            alert("导入图片时出错: " + e);
        }
    }
    
    return placedItems;
}

/**
 * 排列图片
 * @param {Array} items - 图片项数组
 * @param {Object} settings - 排列设置
 */
function arrangeImages(items, settings) {
    var doc = app.activeDocument;
    
    // 使用用户指定的位置
    var startX = settings.positionX;
    var startY = settings.positionY;
    
    // 根据布局类型排列图片
    switch (settings.layoutType) {
        case 0: // 网格布局
            arrangeInGrid(items, startX, startY, settings);
            break;
        case 1: // 单行布局
            arrangeInRow(items, startX, startY, settings);
            break;
        case 2: // 单列布局
            arrangeInColumn(items, startX, startY, settings);
            break;
    }
}

/**
 * 网格布局排列
 * @param {Array} items - 图片项数组
 * @param {number} startX - 起始X坐标
 * @param {number} startY - 起始Y坐标
 * @param {Object} settings - 排列设置
 */
function arrangeInGrid(items, startX, startY, settings) {
    var rows = settings.rows;
    var cols = settings.cols;
    var hSpacing = settings.hSpacing;
    var vSpacing = settings.vSpacing;
    
    // 计算每个图片的最大宽度和高度
    var maxWidth = 0;
    var maxHeight = 0;
    
    for (var i = 0; i < items.length; i++) {
        var bounds = items[i].geometricBounds;
        var width = bounds[2] - bounds[0];
        var height = bounds[1] - bounds[3];
        
        if (width > maxWidth) maxWidth = width;
        if (height > maxHeight) maxHeight = height;
    }
    
    // 排列图片
    for (var i = 0; i < items.length; i++) {
        var row, col;
        if (settings.sortDirection === 0) {
            // 横向排序
            row = Math.floor(i / cols);
            col = i % cols;
        } else {
            // 竖向排序
            col = Math.floor(i / rows);
            row = i % rows;
        }
        
        var x = startX + col * (maxWidth + hSpacing);
        var y = startY - row * (maxHeight + vSpacing);
        
        // 移动图片
        var bounds = items[i].geometricBounds;
        var width = bounds[2] - bounds[0];
        var height = bounds[1] - bounds[3];
        
        // 计算居中偏移
        var offsetX = (maxWidth - width) / 2;
        var offsetY = (maxHeight - height) / 2;
        
        items[i].position = [x + offsetX, y - offsetY];
    }
}

/**
 * 单行布局排列
 * @param {Array} items - 图片项数组
 * @param {number} startX - 起始X坐标
 * @param {number} startY - 起始Y坐标
 * @param {Object} settings - 排列设置
 */
function arrangeInRow(items, startX, startY, settings) {
    var hSpacing = settings.hSpacing;
    
    // 计算每个图片的最大高度
    var maxHeight = 0;
    
    for (var i = 0; i < items.length; i++) {
        var bounds = items[i].geometricBounds;
        var height = bounds[1] - bounds[3];
        
        if (height > maxHeight) maxHeight = height;
    }
    
    // 排列图片
    var currentX = startX;
    
    for (var i = 0; i < items.length; i++) {
        var bounds = items[i].geometricBounds;
        var width = bounds[2] - bounds[0];
        var height = bounds[1] - bounds[3];
        
        // 计算垂直居中偏移
        var offsetY = (maxHeight - height) / 2;
        
        // 移动图片
        items[i].position = [currentX, startY - offsetY];
        
        // 更新下一个图片的X坐标
        currentX += width + hSpacing;
    }
}

/**
 * 单列布局排列
 * @param {Array} items - 图片项数组
 * @param {number} startX - 起始X坐标
 * @param {number} startY - 起始Y坐标
 * @param {Object} settings - 排列设置
 */
function arrangeInColumn(items, startX, startY, settings) {
    var vSpacing = settings.vSpacing;
    
    // 计算每个图片的最大宽度
    var maxWidth = 0;
    
    for (var i = 0; i < items.length; i++) {
        var bounds = items[i].geometricBounds;
        var width = bounds[2] - bounds[0];
        
        if (width > maxWidth) maxWidth = width;
    }
    
    // 排列图片
    var currentY = startY;
    
    for (var i = 0; i < items.length; i++) {
        var bounds = items[i].geometricBounds;
        var width = bounds[2] - bounds[0];
        var height = bounds[1] - bounds[3];
        
        // 计算水平居中偏移
        var offsetX = (maxWidth - width) / 2;
        
        // 移动图片
        items[i].position = [startX + offsetX, currentY];
        
        // 更新下一个图片的Y坐标
        currentY -= height + vSpacing;
    }
}
