const SEND_INTERVAL = 200;
const SEND_QUALITY = 0.6;  

const video = document.getElementById('cam');
const overlay = document.getElementById('overlay');
const capture = document.getElementById('capture');
const startBtn = document.getElementById('startBtn');
const stopBtn = document.getElementById('stopBtn');
const testBtn = document.getElementById('testBtn');
const wsUrlInput = document.getElementById('wsUrl');
const listEl = document.getElementById('list');
const totalEl = document.getElementById('total');
const uniqueEl = document.getElementById('unique');

let ws = null;
let sendIntervalId = null;
let stream = null;

function resizeCanvases() {
    const rect = video.getBoundingClientRect();
    overlay.width = rect.width;
    overlay.height = rect.height;
    capture.width = rect.width;
    capture.height = rect.height;
  }

function drawDetections(detections){
    const ctx = overlay.getContext('2d');
    ctx.clearRect(0,0,overlay.width,overlay.height);
    if(!detections || !detections.length) return;
    detections.forEach(det=>{
      const x = det.x * overlay.width;
      const y = det.y * overlay.height;
      const w = det.w * overlay.width;
      const h = det.h * overlay.height;
      ctx.lineWidth = Math.max(2, Math.round(overlay.width/200));
      ctx.strokeStyle = 'rgba(6,182,212,0.95)';
      ctx.strokeRect(x,y,w,h);
      ctx.fillStyle = 'rgba(6,182,212,0.12)';
      ctx.fillRect(x, y - 22, Math.min(200, 8 + (det.label||'').length * 8), 20);
      ctx.font = '14px system-ui';
      ctx.fillStyle = '#e6eef8';
      const text = `${det.label || 'object'} ${(det.confidence||0).toFixed(2)}`;
      ctx.fillText(text, x+6, y-8);
    })
  }

function updateList(detections){
    const counts = {};
    detections.forEach(d=>{ counts[d.label] = (counts[d.label]||0)+1 });
    listEl.innerHTML = Object.keys(counts).sort((a,b)=>counts[b]-counts[a])
      .map(k=>`<div class="item"><div>${k}</div><div>${counts[k]}</div></div>`).join('') 
      || '<div class="small">No objects</div>';
    totalEl.textContent = detections.length;
    uniqueEl.textContent = Object.keys(counts).length;
  }

async function startCamera(){
    if(stream) return;
    try{
      stream = await navigator.mediaDevices.getUserMedia({video:{facingMode:'environment'}, audio:false});
      video.srcObject = stream;
      await video.play();
      video.addEventListener('loadedmetadata', ()=>{ setTimeout(resizeCanvases,50) });
      window.addEventListener('resize', resizeCanvases);
    }catch(e){
      alert('Camera access denied or not available.');
    }
  }

function stopCamera(){
    if(!stream) return;
    stream.getTracks().forEach(t=>t.stop());
    stream = null;
    video.pause();
    video.srcObject = null;
  }

function startSendingFrames(){
    if(!ws || ws.readyState !== WebSocket.OPEN) return;
    sendIntervalId = setInterval(()=>{
      const ctx = capture.getContext('2d');
      ctx.save();
      ctx.scale(-1,1);
      ctx.drawImage(video, -capture.width, 0, capture.width, capture.height);
      ctx.restore();
      capture.toBlob((blob)=>{
        if(blob && ws.readyState === WebSocket.OPEN){
          ws.send(blob);
        }
      }, 'image/jpeg', SEND_QUALITY);
    }, SEND_INTERVAL);
  }

function stopSendingFrames(){ 
    if(sendIntervalId){ clearInterval(sendIntervalId); sendIntervalId = null } 
  }

function connectWS(url){
    if(ws) ws.close();
    ws = new WebSocket(url);
    ws.binaryType = 'arraybuffer';
    ws.onopen = ()=>{ console.log('ws open'); startSendingFrames(); };
    ws.onmessage = (ev)=>{
      try{
        const data = JSON.parse(ev.data);
        const detections = data.detections || [];
        drawDetections(detections);
        updateList(detections);
      }catch(e){ console.warn('ws parse error', e); }
    };
    ws.onclose = ()=>{ console.log('ws closed'); stopSendingFrames(); }
    ws.onerror = (e)=>{ console.warn('ws err', e); }
  }

function runDemo(){
    const mock = [
      {label:'person', confidence:0.98, x:0.15, y:0.2, w:0.12, h:0.34},
      {label:'bottle', confidence:0.89, x:0.65, y:0.4, w:0.1, h:0.2}
    ];
    drawDetections(mock);
    updateList(mock);
  }

startBtn.addEventListener('click', async ()=>{
    await startCamera();
    const url = wsUrlInput.value.trim();
    if(url) connectWS(url);
  });
  stopBtn.addEventListener('click', ()=>{
    stopSendingFrames(); if(ws) ws.close(); stopCamera(); 
    overlay.getContext('2d').clearRect(0,0,overlay.width,overlay.height); 
    listEl.innerHTML=''; totalEl.textContent='0'; uniqueEl.textContent='0'; 
  });
  testBtn.addEventListener('click', runDemo);

const ro = new ResizeObserver(()=>{ resizeCanvases(); });
ro.observe(document.body);
