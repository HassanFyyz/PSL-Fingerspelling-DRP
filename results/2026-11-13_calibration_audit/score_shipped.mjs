// Score this recording with the actual browser class and frozen model files.
import { readFileSync, writeFileSync } from 'node:fs';
import { pathToFileURL } from 'node:url';
import { resolve } from 'node:path';
const [repoArg,source,out,modelDirArg]=process.argv.slice(2),repo=resolve(repoArg);
const modelDir=resolve(modelDirArg || repo+'/web/model_geom');
const {SvmClassifier}=await import(pathToFileURL(repo+'/web/svm_classifier.js').href);
const man=JSON.parse(readFileSync(modelDir+'/geom_model.json','utf8'));
const f32=n=>{let b=readFileSync(modelDir+'/'+n);return new Float32Array(b.buffer,b.byteOffset,b.byteLength/4)};
const m=new SvmClassifier(man,f32('geom_sv.bin'),f32('geom_dualcoef.bin'),f32('geom_intercept.bin'),f32('geom_scaler_mean.bin'),f32('geom_scaler_scale.bin'));
const d=JSON.parse(readFileSync(source,'utf8')),rows=[];
for(const [i,t] of d.takes.entries()){
 const v=m.predict(t.frames.map(f=>f.pts));let best=0;for(let j=1;j<v.length;j++)if(v[j]>v[best])best=j;
 rows.push({letter:t.letter,take:t.take,handedness:t.handedness,n_frames:t.frames.length,known_class:man.classes.includes(t.letter),pred:man.classes[best],correct:man.classes[best]===t.letter,top3:m.top(v,3),vote_shares:Array.from(v)});
 if((i+1)%50===0)console.log(`${i+1}/${d.takes.length} takes`);
}
writeFileSync(out,JSON.stringify({source,model:modelDir,train_holds:man.train_holds,score:'mean OVO vote share over all take frames, exactly as SvmClassifier.predict',rows},null,2));
const known=rows.filter(r=>r.known_class);console.log('Frozen browser model:',known.filter(r=>r.correct).length,'/',known.length);
