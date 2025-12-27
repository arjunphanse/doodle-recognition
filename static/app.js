document.addEventListener("DOMContentLoaded", () => {
    // Elements
    const canvas = document.getElementById("canvas")
    const ctx = canvas.getContext("2d")
    // Buttons
    const clearBtn = document.getElementById("clear-btn")
    const predictBtn = document.getElementById("predict-btn")
    // Predictions
    const predictionsConv = document.getElementById("predictions-conv")
    const predictionsRes = document.getElementById("predictions-res")
    const predictionsDino = document.getElementById("predictions-dino")
    const barConv = document.getElementById("bar-conv")
    const barRes = document.getElementById("bar-res")
    const barDino = document.getElementById("bar-dino")
    // Classes list
    const classesToggle = document.getElementById("classes-toggle")
    const classesList = document.getElementById("classes-list")
    const classesUl = document.getElementById("classes-ul")
    // 28x28 preview
    const preview = document.getElementById("preview")
    const pctx = preview ? preview.getContext("2d") : null

    // Data
    const COLORS = {
        conv: "#2563eb",
        res: "#059669",
        dino: "#8b5cf6",
    }
    // State
    let drawing = false

    function updateTopBar(bar, color, entry) {
        if (!bar) return
        if (!entry) {
            bar.style.width = "0%"
            bar.style.backgroundColor = ""
            return
        }
        const pct = (entry.prob * 100).toFixed(1)
        bar.style.width = `${pct}%`
        bar.style.backgroundColor = color
    }

    function renderList(target, items, pctClass) {
        target.innerHTML = ""
        for (const p of items) {
            const li = document.createElement("li")
            const name = document.createElement("span")
            name.textContent = p.class
            const pct = document.createElement("span")
            pct.className = `pct ${pctClass}`
            pct.textContent = `${(p.prob * 100).toFixed(2)}%`
            li.appendChild(name)
            li.appendChild(document.createTextNode(" "))
            li.appendChild(pct)
            target.appendChild(li)
        }
    }

    function resetPredictions() {
        predictionsConv.innerHTML = ""
        predictionsRes.innerHTML = ""
        predictionsDino.innerHTML = ""
        updateTopBar(barConv, "", null)
        updateTopBar(barRes, "", null)
        updateTopBar(barDino, "", null)
    }

    function setWhiteBackground() {
        ctx.fillStyle = "#ffffff"
        ctx.fillRect(0, 0, canvas.width, canvas.height)
    }

    function startDraw(e) {
        drawing = true
        const pos = getPos(e)
        ctx.beginPath()
        ctx.moveTo(pos.x, pos.y)
    }

    function draw(e) {
        if (!drawing) return
        const pos = getPos(e)
        ctx.lineWidth = 9
        ctx.lineCap = "round"
        ctx.strokeStyle = "#000000"
        ctx.lineTo(pos.x, pos.y)
        ctx.stroke()
    }

    function endDraw() {
        drawing = false
        ctx.beginPath()
    }

    function getPos(e) {
        const rect = canvas.getBoundingClientRect()
        const clientX = e.touches ? e.touches[0].clientX : e.clientX
        const clientY = e.touches ? e.touches[0].clientY : e.clientY
        return { x: clientX - rect.left, y: clientY - rect.top }
    }

    function clearCanvas() {
        setWhiteBackground()
        resetPredictions()
        if (pctx) {
            pctx.fillStyle = "#ffffff"
            pctx.fillRect(0, 0, 28, 28)
        }
    }

    async function predict() {
        const small = document.createElement("canvas")
        small.width = 28
        small.height = 28
        const sctx = small.getContext("2d")
        sctx.fillStyle = "#ffffff"
        sctx.fillRect(0, 0, 28, 28)
        sctx.drawImage(canvas, 0, 0, 28, 28)
        if (pctx) {
            pctx.imageSmoothingEnabled = false
            pctx.clearRect(0, 0, 28, 28)
            pctx.drawImage(small, 0, 0)
        }
        const data = sctx.getImageData(0, 0, 28, 28).data
        const pixels = []
        for (let i = 0; i < data.length; i += 4) {
            pixels.push(data[i])
        }

        const res = await fetch("/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ pixels }),
        })

        if (!res.ok) {
            const errMsg = `<li>Error: ${res.statusText}</li>`
            predictionsConv.innerHTML = errMsg
            predictionsRes.innerHTML = errMsg
            predictionsDino.innerHTML = errMsg
            return
        }

        const json = await res.json()
        const conv = (json.convnet || []).slice(0, 5)
        const resnet = (json.resnet || []).slice(0, 5)
        const dino = (json.dinov2 || []).slice(0, 5)
        updateTopBar(barConv, COLORS.conv, conv[0])
        updateTopBar(barRes, COLORS.res, resnet[0])
        updateTopBar(barDino, COLORS.dino, dino[0])

        renderList(predictionsConv, conv, "pct-conv")
        renderList(predictionsRes, resnet, "pct-res")
        renderList(predictionsDino, dino, "pct-dino")
    }

    // Event listeners
    canvas.addEventListener("mousedown", startDraw)
    canvas.addEventListener("mousemove", draw)
    canvas.addEventListener("mouseup", endDraw)
    canvas.addEventListener("mouseleave", endDraw)
    canvas.addEventListener("touchstart", (e) => {
        startDraw(e)
        e.preventDefault()
    })
    canvas.addEventListener("touchmove", (e) => {
        draw(e)
        e.preventDefault()
    })
    canvas.addEventListener("touchend", endDraw)

    clearBtn.addEventListener("click", clearCanvas)
    predictBtn.addEventListener("click", predict)

    // Populate classes list
    CLASSES.forEach((c) => {
        const li = document.createElement("li")
        li.textContent = c
        classesUl.appendChild(li)
    })

    classesToggle.addEventListener("click", () => {
        const isHidden = classesList.classList.contains("hidden")
        classesList.classList.toggle("hidden")
        classesToggle.textContent = isHidden ? "Hide class list" : "Show class list"
    })

    setWhiteBackground()
})
