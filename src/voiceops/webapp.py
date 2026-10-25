"""Browser demo: talk to the agent with your microphone (Gnani STT -> agent -> Gnani TTS), or type.

    python run.py serve   ->   http://127.0.0.1:8000
"""
from __future__ import annotations

import tempfile
import uuid
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

from .agent import SupportAgent
from .config import settings

PAGE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ZentroMart VoiceOps</title>
<style>
:root{--bg:#f6f5f1;--card:#fff;--ink:#1d1d1b;--mute:#6b6a64;--line:#e4e2da;--acc:#0f6e56;--acc2:#e8f3ef;--user:#1d1d1b}
@media (prefers-color-scheme:dark){:root{--bg:#161614;--card:#1f1f1c;--ink:#f1efe8;--mute:#9a988f;--line:#33322d;--acc:#4fb996;--acc2:#1d2b26;--user:#f1efe8}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
.wrap{max-width:1100px;margin:0 auto;padding:20px 16px;display:grid;grid-template-columns:1fr 340px;gap:16px}
@media(max-width:860px){.wrap{grid-template-columns:1fr}}
h1{font-size:18px;margin:0 0 2px}p.sub{margin:0 0 14px;color:var(--mute);font-size:13px}
.card{background:var(--card);border:1px solid var(--line);border-radius:14px}
#log{height:56vh;overflow:auto;padding:14px;display:flex;flex-direction:column;gap:10px}
.msg{max-width:82%;padding:9px 13px;border-radius:14px;white-space:pre-wrap}
.me{align-self:flex-end;background:var(--user);color:var(--bg)}.bot{align-self:flex-start;background:var(--acc2)}
.heard{font-size:11px;opacity:.7;margin-top:2px}
.bar{display:flex;gap:8px;padding:12px;border-top:1px solid var(--line)}
input[type=text]{flex:1;padding:10px 12px;border-radius:10px;border:1px solid var(--line);background:transparent;color:var(--ink);font:inherit}
button{border:0;border-radius:10px;padding:10px 14px;font:inherit;cursor:pointer;background:var(--acc);color:#fff}
button.ghost{background:transparent;color:var(--ink);border:1px solid var(--line)}
button.rec{background:#c0392b;animation:p 1s infinite}@keyframes p{50%{opacity:.6}}
.side{padding:14px}.side h3{margin:0 0 6px;font-size:12px;text-transform:uppercase;letter-spacing:.06em;color:var(--mute)}
.chip{display:inline-block;margin:0 4px 4px 0;padding:2px 8px;border-radius:99px;background:var(--acc2);color:var(--acc);font:12px ui-monospace,monospace}
.tool{font:12px ui-monospace,monospace;margin:0 0 4px;color:var(--mute);word-break:break-all}
.sec{margin-bottom:14px}.mono{font:12px ui-monospace,monospace;color:var(--mute)}
</style></head><body><div class="wrap"><div>
<h1>ZentroMart VoiceOps</h1><p class="sub">Hold the mic button, speak, release. The agent listens with Gnani STT and answers with Gnani TTS.</p>
<div class="card"><div id="log"></div>
<div class="bar"><button id="mic" class="ghost">🎙 Hold to talk</button><input id="txt" type="text" placeholder="…or type a message and press Enter"><button id="send">Send</button></div></div>
<audio id="player"></audio></div>
<div class="card side"><div class="sec"><h3>Evidence used (chunk IDs)</h3><div id="src" class="mono">–</div></div>
<div class="sec"><h3>Tool calls</h3><div id="tools" class="mono">–</div></div>
<div class="sec"><h3>Retrieval query</h3><div id="rq" class="mono">–</div></div>
<div class="sec"><h3>Latency</h3><div id="lat" class="mono">–</div></div>
<button class="ghost" id="reset">New conversation</button></div></div>
<script>
let sid=crypto.randomUUID();const $=id=>document.getElementById(id);
function add(cls,text,sub){const d=document.createElement('div');d.className='msg '+cls;d.textContent=text;if(sub){const s=document.createElement('div');s.className='heard';s.textContent=sub;d.appendChild(s)}$('log').appendChild(d);$('log').scrollTop=1e9;return d}
function show(r){$('src').innerHTML=(r.sources||[]).map(s=>'<span class="chip">'+s+'</span>').join('')||'–';
$('tools').innerHTML=(r.tool_calls||[]).map(t=>'<div class="tool">'+t.name+'('+JSON.stringify(t.args)+')</div>').join('')||'–';
$('rq').textContent=r.rewritten_query||'–';$('lat').textContent=r.latency?Object.entries(r.latency).map(([k,v])=>k+' '+v+'s').join(' · '):'–'}
async function turn(body,isForm,label){const w=add('me',label||'…');
 try{const res=await fetch('/api/turn',{method:'POST',headers:isForm?{}:{'Content-Type':'application/json'},body:isForm?body:JSON.stringify(body)});const r=await res.json();
 if(r.heard!==undefined&&isForm){w.textContent='🎙 '+(r.heard||'(nothing heard)')}
 add('bot',r.reply||r.error,r.ticket_id?'ticket '+r.ticket_id:'');show(r);
 if(r.audio_url){const p=$('player');p.src=r.audio_url;p.play().catch(()=>{})}}catch(e){add('bot','Error: '+e)}}
$('send').onclick=send;$('txt').onkeydown=e=>{if(e.key==='Enter')send()};
function send(){const t=$('txt').value.trim();if(!t)return;$('txt').value='';turn({session_id:sid,text:t},false,t)}
$('reset').onclick=()=>{sid=crypto.randomUUID();$('log').innerHTML='';show({})};
// ---- microphone -> 16 kHz mono WAV (no ffmpeg needed)
let ctx,stream,proc,chunks=[],recording=false;
async function start(){if(recording)return;stream=await navigator.mediaDevices.getUserMedia({audio:{echoCancellation:true,noiseSuppression:true}});
 ctx=new AudioContext();const src=ctx.createMediaStreamSource(stream);proc=ctx.createScriptProcessor(4096,1,1);chunks=[];
 proc.onaudioprocess=e=>chunks.push(new Float32Array(e.inputBuffer.getChannelData(0)));src.connect(proc);proc.connect(ctx.destination);recording=true;$('mic').classList.add('rec');$('mic').textContent='● Recording…'}
async function stop(){if(!recording)return;recording=false;$('mic').classList.remove('rec');$('mic').textContent='🎙 Hold to talk';
 proc.disconnect();stream.getTracks().forEach(t=>t.stop());const rate=ctx.sampleRate;await ctx.close();
 const n=chunks.reduce((a,c)=>a+c.length,0);const all=new Float32Array(n);let o=0;chunks.forEach(c=>{all.set(c,o);o+=c.length});
 if(n<rate*0.3){return}const wav=toWav(downsample(all,rate,16000),16000);
 const fd=new FormData();fd.append('session_id',sid);fd.append('audio',new Blob([wav],{type:'audio/wav'}),'turn.wav');turn(fd,true,'🎙 …')}
function downsample(b,from,to){if(from===to)return b;const r=from/to,l=Math.floor(b.length/r),o=new Float32Array(l);for(let i=0;i<l;i++){const s=Math.floor(i*r),e=Math.min(Math.floor((i+1)*r),b.length);let a=0;for(let j=s;j<e;j++)a+=b[j];o[i]=a/(e-s||1)}return o}
function toWav(f,rate){const buf=new ArrayBuffer(44+f.length*2),v=new DataView(buf);const w=(o,s)=>{for(let i=0;i<s.length;i++)v.setUint8(o+i,s.charCodeAt(i))};
 w(0,'RIFF');v.setUint32(4,36+f.length*2,true);w(8,'WAVEfmt ');v.setUint32(16,16,true);v.setUint16(20,1,true);v.setUint16(22,1,true);v.setUint32(24,rate,true);v.setUint32(28,rate*2,true);v.setUint16(32,2,true);v.setUint16(34,16,true);w(36,'data');v.setUint32(40,f.length*2,true);
 for(let i=0;i<f.length;i++){const s=Math.max(-1,Math.min(1,f[i]));v.setInt16(44+i*2,s<0?s*0x8000:s*0x7fff,true)}return buf}
const m=$('mic');m.onmousedown=start;m.onmouseup=stop;m.onmouseleave=stop;m.ontouchstart=e=>{e.preventDefault();start()};m.ontouchend=stop;
</script></body></html>"""


def create_app(agent_obj=None, voice_obj=None) -> Flask:
    app = Flask(__name__)
    tmp = Path(tempfile.mkdtemp(prefix="voiceops_"))
    state: dict = {"agent": agent_obj, "voice": voice_obj, "sessions": {}}

    def agent() -> SupportAgent:
        if state["agent"] is None:
            state["agent"] = SupportAgent()
        return state["agent"]

    def voice():
        if state["voice"] is None and settings.gnani_api_key:
            from .voice import GnaniVoice

            state["voice"] = GnaniVoice()
        return state["voice"]

    @app.get("/")
    def index():
        return PAGE

    @app.post("/api/turn")
    def turn():
        ag = agent()
        if request.files.get("audio"):
            sid = request.form.get("session_id", "web")
            sess = state["sessions"].setdefault(sid, ag.new_session(sid, "voice"))
            v = voice()
            if v is None:
                return jsonify({"error": "GNANI_API_KEY is not set, so voice is unavailable. Type instead."}), 400
            f = tmp / f"{uuid.uuid4().hex}.wav"
            request.files["audio"].save(f)
            stt = v.transcribe(f)
            if not stt.text.strip():
                from .voice import REPROMPT

                reply, res = REPROMPT["en-IN"], None
                out = {"heard": "", "reply": reply, "sources": [], "tool_calls": [], "latency": {"stt": stt.seconds}}
            else:
                res = ag.respond(sess, stt.text, channel="voice", alternatives=stt.alternatives)
                out = {"heard": stt.text, "reply": res.reply, "sources": res.sources, "tool_calls": res.tool_calls,
                       "ticket_id": res.ticket_id, "rewritten_query": res.rewritten_query,
                       "latency": {"stt": stt.seconds, "agent": res.latency_s}}
            wav = tmp / f"{uuid.uuid4().hex}.reply.wav"
            tr = v.synthesize(out["reply"], wav)
            if tr.path:
                out["audio_url"] = f"/api/audio/{wav.name}"
            else:
                out["tts_error"] = tr.error
            return jsonify(out)
        data = request.get_json(force=True)
        sid = data.get("session_id", "web")
        sess = state["sessions"].setdefault(sid, ag.new_session(sid, "text"))
        res = ag.respond(sess, data.get("text", ""), channel="text")
        out = {"reply": res.reply, "sources": res.sources, "tool_calls": res.tool_calls, "ticket_id": res.ticket_id,
               "rewritten_query": res.rewritten_query, "latency": {"agent": res.latency_s}}
        v = voice()
        if v is not None and data.get("speak"):
            wav = tmp / f"{uuid.uuid4().hex}.reply.wav"
            tr = v.synthesize(res.reply, wav)
            if tr.path:
                out["audio_url"] = f"/api/audio/{wav.name}"
        return jsonify(out)

    @app.get("/api/audio/<name>")
    def audio(name):
        return send_from_directory(tmp, name, mimetype="audio/wav")

    return app
