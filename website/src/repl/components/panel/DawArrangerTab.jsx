import cx from '@src/cx.mjs';
import { useState, useEffect, useRef } from 'react';
import { WebMidi, enableWebMidi } from '@strudel/midi';

// Standard 13-Channel MIDI Map for Bitwig / DAW Studio integration
const CHANNELS = [
  { id: 'kick', ch: 1, name: 'Kick', defaultNote: 36, color: '#FF3B3B', type: 'punchcard' },
  { id: 'hats', ch: 2, name: 'Hats', defaultNote: 42, color: '#3BFFB8', type: 'punchcard' },
  { id: 'tops', ch: 3, name: 'Tops / Accents', defaultNote: 46, color: '#3BE1FF', type: 'punchcard' },
  { id: 'perc', ch: 4, name: 'Perc / Shaker', defaultNote: 70, color: '#FFD93B', type: 'punchcard' },
  { id: 'snareClap', ch: 5, name: 'Clap / Snare', defaultNote: 39, color: '#FF3BE1', type: 'punchcard' },
  { id: 'ride', ch: 6, name: 'Ride / Cymbal', defaultNote: 37, color: '#FF8A3B', type: 'punchcard' },
  { id: 'bass', ch: 7, name: 'Bass', defaultNote: 36, color: '#B83BFF', type: 'pianoroll' },
  { id: 'subBass', ch: 8, name: 'Sub Bass', defaultNote: 24, color: '#8A3BFF', type: 'pianoroll' },
  { id: 'chords', ch: 9, name: 'Chords', defaultNote: 60, color: '#3BFF57', type: 'pianoroll' },
  { id: 'pad', ch: 10, name: 'Pad', defaultNote: 48, color: '#FF6B3B', type: 'pianoroll' },
  { id: 'arp', ch: 11, name: 'Arp / Lead', defaultNote: 60, color: '#FFEE3B', type: 'pianoroll' },
  { id: 'fxRoll', ch: 12, name: 'FX Roll / Buildup', defaultNote: 38, color: '#FFB03B', type: 'punchcard' },
  { id: 'marker', ch: 13, name: 'Section Marker', defaultNote: 96, color: '#FFFFFF', type: 'punchcard' },
];

const NOTE = { kick: 36, hat: 42, openHat: 46, perc: 70, clap: 39, snare: 38, ride: 37 };

function bjorklund(pulses, steps) {
  if (pulses <= 0) return Array(steps).fill(false);
  if (pulses >= steps) return Array(steps).fill(true);
  let counts = [],
    remainders = [pulses];
  let divisor = steps - pulses,
    level = 0;
  while (true) {
    counts.push(Math.floor(divisor / remainders[level]));
    remainders.push(divisor % remainders[level]);
    divisor = remainders[level];
    level++;
    if (remainders[level] <= 1) break;
  }
  counts.push(divisor);
  let pattern = [];
  function build(lvl) {
    if (lvl === -1) pattern.push(0);
    else if (lvl === -2) pattern.push(1);
    else {
      for (let i = 0; i < counts[lvl]; i++) build(lvl - 1);
      if (remainders[lvl] !== 0) build(lvl - 2);
    }
  }
  build(level);
  const first = pattern.indexOf(1);
  pattern = pattern.slice(first).concat(pattern.slice(0, first));
  return pattern.map(Boolean);
}

// Rhythm primitives
function kick4Floor(vel = 100) {
  return [0, 1, 2, 3].map((b) => ({ beat: b, dur: 0.25, note: NOTE.kick, vel }));
}
function kickHalfTime(vel = 100) {
  return [{ beat: 0, dur: 0.3, note: NOTE.kick, vel }];
}
function kickBreakbeat(vel = 100) {
  return [0, 2.5].map((b) => ({ beat: b, dur: 0.3, note: NOTE.kick, vel }));
}
function hatsOffbeat(vel = 75) {
  return [0.5, 1.5, 2.5, 3.5].map((b) => ({ beat: b, dur: 0.15, note: NOTE.hat, vel }));
}
function hatsQuarterOffbeat(vel = 80) {
  return [1, 3].map((b) => ({ beat: b, dur: 0.15, note: NOTE.hat, vel }));
}
function hats16thShuffled(vel = 60) {
  const v = [90, 40, 60, 40, 85, 40, 65, 40, 90, 40, 55, 40, 85, 40, 70, 40];
  return Array.from({ length: 16 }, (_, i) => ({
    beat: i * 0.25,
    dur: 0.1,
    note: NOTE.hat,
    vel: Math.round((v[i] * vel) / 70),
  }));
}
function hatsSwingGroove(vel = 55) {
  return [0, 0.55, 1.5, 2.05, 2.55, 3.5].map((b) => ({ beat: b, dur: 0.12, note: NOTE.hat, vel }));
}
function topsOpen(vel = 60) {
  return [0.5, 1.5, 2.5, 3.5].map((b) => ({ beat: b, dur: 0.2, note: NOTE.openHat, vel }));
}
function topsSparse(vel = 50) {
  return [1.5, 3.5].map((b) => ({ beat: b, dur: 0.2, note: NOTE.openHat, vel }));
}
function topsAccent(vel = 50) {
  return [{ beat: 3.75, dur: 0.15, note: NOTE.openHat, vel }];
}
function percSteady8th(vel = 45) {
  return [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5].map((b) => ({ beat: b, dur: 0.1, note: NOTE.perc, vel }));
}
function clapBackbeat(vel = 90) {
  return [1, 3].map((b) => ({ beat: b, dur: 0.15, note: NOTE.clap, vel }));
}
function clapSparse(vel = 70) {
  return [{ beat: 2, dur: 0.15, note: NOTE.clap, vel }];
}
function snareBackbeatSingle(vel = 110) {
  return [{ beat: 2, dur: 0.2, note: NOTE.snare, vel }];
}
function rideOffKick(vel = 55) {
  return [{ beat: 2, dur: 0.15, note: NOTE.ride, vel }];
}
function rideSyncopated(vel = 40) {
  return [1, 3.5].map((b) => ({ beat: b, dur: 0.15, note: NOTE.ride, vel }));
}
function bassEuclidRolling(vals, pulses = 5, steps = 8, vel = 85) {
  const mask = bjorklund(pulses, steps),
    stepDur = 4 / steps,
    out = [];
  for (let i = 0; i < steps; i++) {
    if (mask[i]) out.push({ beat: i * stepDur, dur: stepDur * 0.8, note: vals[i % vals.length], vel });
  }
  return out;
}
function bassStraight16(root, vel = 75) {
  return Array.from({ length: 16 }, (_, i) => ({
    beat: i * 0.25,
    dur: 0.2,
    note: root,
    vel: i % 4 === 0 ? vel + 10 : vel,
  }));
}
function bassLoop8th(root, vel = 85) {
  return [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5].map((b) => ({ beat: b, dur: 0.4, note: root, vel }));
}
function bassWobble(rootHi, rootLo, vel = 100) {
  const vals = [rootHi, rootLo, rootHi, rootLo, rootHi, rootLo, rootHi, rootLo];
  return [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5].map((b, i) => ({ beat: b, dur: 0.4, note: vals[i], vel }));
}
function bassGroove(vals, vel = 70) {
  const beats = [0, 0.75, 1.5, 2, 2.75, 3.5];
  return beats.map((b, i) => ({ beat: b, dur: 0.5, note: vals[i % vals.length], vel }));
}
function subReggae(root, vel = 100) {
  return [0.75, 2.75].map((b) => ({ beat: b, dur: 1.2, note: root, vel }));
}
function subOffset(root, vel = 90) {
  return [1.25, 3.25].map((b) => ({ beat: b, dur: 1.0, note: root, vel }));
}
function subQuarter(vals, vel = 90) {
  return vals.map((n, i) => ({ beat: i, dur: 0.9, note: n, vel }));
}
function subSustain(root, vel = 90) {
  return [
    { beat: 0, dur: 1.9, note: root, vel },
    { beat: 2, dur: 1.9, note: root, vel },
  ];
}
function fxRoll16(vel = 55) {
  return Array.from({ length: 16 }, (_, i) => ({ beat: i * 0.25, dur: 0.1, note: NOTE.snare, vel }));
}
function fxRoll8(vel = 40) {
  return [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5].map((b) => ({ beat: b, dur: 0.15, note: NOTE.snare, vel }));
}

let chordCounter = 0;
let padCounter = 0;
function cycleChordHits(voicings, beat, dur, vel) {
  const chord = voicings[chordCounter % voicings.length];
  chordCounter++;
  return chord.map((n) => ({ beat, dur, note: n, vel }));
}
function cyclePadHits(notes, dur, vel) {
  const n = notes[padCounter % notes.length];
  padCounter++;
  return [{ beat: 0, dur, note: n, vel }];
}
function arpRepeat(vals, times, vel) {
  const seq = [];
  for (let r = 0; r < times; r++) vals.forEach((n) => seq.push(n));
  const stepDur = 4 / seq.length;
  return seq.map((n, i) => ({ beat: i * stepDur, dur: stepDur * 0.8, note: n, vel }));
}
function arpSparse(vals, vel) {
  return [0.5, 1.5, 2.5, 3.5].map((b, i) => ({ beat: b, dur: 0.4, note: vals[i % vals.length], vel }));
}

const GENRES = {
  progHouse: {
    label: 'Progressive House',
    bpm: 130,
    roles: {
      kick: () => kick4Floor(100),
      hats: () => hatsQuarterOffbeat(80),
      tops: () => topsOpen(60),
      perc: () => percSteady8th(45),
      snareClap: () => clapBackbeat(90),
      ride: () => rideOffKick(55),
      bass: () => bassEuclidRolling([36, 36, 39, 41], 5, 8, 85),
      subBass: () => subQuarter([24, 24, 27, 29], 90),
      chords: () =>
        cycleChordHits(
          [
            [48, 51, 55],
            [48, 51, 55],
            [44, 48, 51],
            [41, 44, 48],
          ],
          2,
          1.8,
          65,
        ),
      pad: () => cyclePadHits([36, 39, 32, 41], 3.9, 35),
      arp: () => arpRepeat([48, 51, 55, 60], 2, 55),
      fxRoll: () => fxRoll16(55),
    },
    sections: [
      { name: 'Intro', bars: 8, active: ['kick', 'hats', 'perc'] },
      { name: 'BuildUp', bars: 8, active: ['kick', 'hats', 'tops', 'perc', 'snareClap', 'bass', 'subBass'] },
      { name: 'Breakdown', bars: 8, active: ['pad', 'chords', 'arp'] },
      { name: 'RiserUp', bars: 4, active: ['pad', 'chords', 'fxRoll', 'marker'] },
      {
        name: 'Drop',
        bars: 16,
        active: ['kick', 'hats', 'tops', 'perc', 'snareClap', 'ride', 'bass', 'subBass', 'chords', 'pad', 'arp'],
      },
      { name: 'Outro', bars: 8, active: ['kick', 'hats', 'bass', 'pad'] },
    ],
  },
  techno: {
    label: 'Techno',
    bpm: 132,
    roles: {
      kick: () => kick4Floor(100),
      hats: () => hatsOffbeat(70),
      tops: () => topsAccent(50),
      perc: () => percSteady8th(30),
      snareClap: () => clapSparse(70),
      ride: () => rideSyncopated(40),
      bass: () => bassLoop8th(36, 90),
      subBass: () => subSustain(24, 70),
      chords: () => cycleChordHits([[60], [63], [60], [58]], 2.5, 0.5, 40),
      pad: () => cyclePadHits([48, 51], 3.9, 25),
      arp: () => arpRepeat([60, 63, 67, 70], 4, 45),
      fxRoll: () => fxRoll16(50),
    },
    sections: [
      { name: 'Intro', bars: 8, active: ['kick', 'hats', 'perc'] },
      { name: 'Build', bars: 16, active: ['kick', 'hats', 'tops', 'perc', 'snareClap', 'bass', 'subBass', 'chords'] },
      { name: 'Break', bars: 8, active: ['pad', 'chords', 'arp', 'fxRoll'] },
      { name: 'RiserUp', bars: 4, active: ['pad', 'fxRoll', 'marker'] },
      {
        name: 'MainLoop',
        bars: 24,
        active: ['kick', 'hats', 'tops', 'perc', 'snareClap', 'ride', 'bass', 'subBass', 'chords', 'pad', 'arp'],
      },
      { name: 'Outro', bars: 8, active: ['kick', 'hats', 'bass', 'pad'] },
    ],
  },
  trance: {
    label: 'Trance',
    bpm: 138,
    roles: {
      kick: () => kick4Floor(100),
      hats: () => hatsOffbeat(75),
      tops: () => topsOpen(55),
      perc: () => percSteady8th(35),
      snareClap: () => clapBackbeat(95),
      ride: () => rideOffKick(40),
      bass: () => bassStraight16(36, 80),
      subBass: () => subQuarter([24, 24, 27, 29], 85),
      chords: () =>
        cycleChordHits(
          [
            [60, 64, 67],
            [60, 64, 67],
            [57, 60, 64],
            [55, 60, 64],
          ],
          0,
          2,
          60,
        ),
      pad: () => cyclePadHits([48, 52, 55, 53], 3.9, 40),
      arp: () => arpRepeat([60, 64, 67, 72], 4, 50),
      fxRoll: () => fxRoll16(60),
    },
    sections: [
      { name: 'Intro', bars: 8, active: ['kick', 'hats', 'perc'] },
      { name: 'Build', bars: 16, active: ['kick', 'hats', 'tops', 'perc', 'snareClap', 'bass', 'subBass'] },
      { name: 'Breakdown', bars: 16, active: ['pad', 'chords', 'arp'] },
      { name: 'RiserUp', bars: 8, active: ['pad', 'chords', 'fxRoll', 'marker'] },
      {
        name: 'Drop',
        bars: 16,
        active: ['kick', 'hats', 'tops', 'perc', 'snareClap', 'ride', 'bass', 'subBass', 'chords', 'pad', 'arp'],
      },
      { name: 'Outro', bars: 8, active: ['kick', 'hats', 'bass', 'pad'] },
    ],
  },
  dnb: {
    label: 'Drum & Bass',
    bpm: 174,
    roles: {
      kick: () => kickBreakbeat(100),
      hats: () => hats16thShuffled(60),
      tops: () => topsSparse(55),
      perc: () => percSteady8th(15),
      snareClap: () => snareBackbeatSingle(110),
      ride: () => [],
      bass: () => subReggae(24, 100),
      subBass: () => subOffset(22, 90),
      chords: () =>
        cycleChordHits(
          [
            [58, 61, 65],
            [55, 58, 62],
            [53, 56, 60],
            [51, 55, 58],
          ],
          0,
          0.6,
          50,
        ),
      pad: () => cyclePadHits([46, 50, 43, 48], 3.9, 30),
      arp: () => arpSparse([58, 61, 65, 68], 45),
      fxRoll: () => fxRoll16(55),
    },
    sections: [
      { name: 'Intro', bars: 8, active: ['kick', 'hats', 'perc'] },
      { name: 'Build', bars: 8, active: ['kick', 'hats', 'tops', 'perc', 'snareClap', 'bass'] },
      {
        name: 'Drop1',
        bars: 16,
        active: ['kick', 'hats', 'tops', 'perc', 'snareClap', 'bass', 'subBass', 'chords', 'arp'],
      },
      { name: 'Breakdown', bars: 8, active: ['pad', 'chords', 'arp'] },
      { name: 'RiserUp', bars: 4, active: ['pad', 'fxRoll', 'marker'] },
      {
        name: 'Drop2',
        bars: 16,
        active: ['kick', 'hats', 'tops', 'perc', 'snareClap', 'bass', 'subBass', 'chords', 'pad', 'arp'],
      },
      { name: 'Outro', bars: 8, active: ['kick', 'hats', 'bass', 'pad'] },
    ],
  },
  dubstep: {
    label: 'Dubstep',
    bpm: 140,
    roles: {
      kick: () => kickHalfTime(100),
      hats: () => hatsSwingGroove(50),
      tops: () => [],
      perc: () => percSteady8th(15),
      snareClap: () => snareBackbeatSingle(115),
      ride: () => [],
      bass: () => bassWobble(36, 24, 105),
      subBass: () => subSustain(24, 90),
      chords: () =>
        cycleChordHits(
          [
            [58, 61, 65],
            [55, 58, 62],
            [53, 56, 60],
            [51, 55, 58],
          ],
          0,
          0.5,
          55,
        ),
      pad: () => cyclePadHits([46, 50, 43, 48], 3.9, 30),
      arp: () => arpSparse([58, 61, 65, 70], 40),
      fxRoll: () => fxRoll16(55),
    },
    sections: [
      { name: 'Intro', bars: 8, active: ['kick', 'hats', 'perc'] },
      { name: 'Build', bars: 8, active: ['kick', 'hats', 'perc', 'snareClap', 'bass'] },
      { name: 'Drop1', bars: 16, active: ['kick', 'hats', 'perc', 'snareClap', 'bass', 'subBass', 'chords'] },
      { name: 'Breakdown', bars: 8, active: ['pad', 'chords', 'arp'] },
      { name: 'RiserUp', bars: 4, active: ['pad', 'fxRoll', 'marker'] },
      { name: 'Drop2', bars: 16, active: ['kick', 'hats', 'perc', 'snareClap', 'bass', 'subBass', 'chords', 'pad'] },
      { name: 'Outro', bars: 8, active: ['kick', 'hats', 'bass', 'pad'] },
    ],
  },
  deepHouse: {
    label: 'Deep House',
    bpm: 122,
    roles: {
      kick: () => kick4Floor(80),
      hats: () => hatsSwingGroove(55),
      tops: () => topsSparse(45),
      perc: () => percSteady8th(25),
      snareClap: () => clapBackbeat(80),
      ride: () => rideSyncopated(35),
      bass: () => bassGroove([36, 39, 41, 43], 70),
      subBass: () => subQuarter([24, 24, 27, 29], 60),
      chords: () =>
        cycleChordHits(
          [
            [60, 64, 67, 71],
            [57, 60, 64, 67],
            [62, 65, 69, 72],
            [55, 59, 62, 65],
          ],
          2.5,
          1,
          55,
        ),
      pad: () => cyclePadHits([48, 52, 55, 53], 3.9, 28),
      arp: () => arpSparse([60, 64, 67, 71], 40),
      fxRoll: () => fxRoll8(40),
    },
    sections: [
      { name: 'Intro', bars: 8, active: ['kick', 'hats', 'perc'] },
      { name: 'Build', bars: 8, active: ['kick', 'hats', 'tops', 'perc', 'snareClap', 'bass'] },
      {
        name: 'Groove1',
        bars: 16,
        active: ['kick', 'hats', 'tops', 'perc', 'snareClap', 'ride', 'bass', 'subBass', 'chords'],
      },
      { name: 'Breakdown', bars: 8, active: ['pad', 'chords', 'arp'] },
      { name: 'RiserUp', bars: 4, active: ['pad', 'fxRoll', 'marker'] },
      {
        name: 'Groove2',
        bars: 16,
        active: ['kick', 'hats', 'tops', 'perc', 'snareClap', 'ride', 'bass', 'subBass', 'chords', 'pad', 'arp'],
      },
      { name: 'Outro', bars: 8, active: ['kick', 'hats', 'bass', 'pad'] },
    ],
  },
};

const STRUDEL_PATTERNS = {
  progHouse: (device, viz) => `/* 13-Channel DAW Studio Arrangement — Progressive House */
setcpm(130/4) // 130 bpm

const kick = s("bd*4").note(36).gain(1)${viz.kick}.midichan(1)
const hats = s("~ hh ~ hh").note(42)${viz.hats}.midichan(2)
const tops = s("~ oh ~ oh ~ oh ~ oh").note(46)${viz.tops}.midichan(3)
const perc = s("shaker*8").note(70).gain(0.35)${viz.perc}.midichan(4)
const snareClap = s("~ cp ~ cp").note(39).gain(0.9)${viz.snareClap}.midichan(5)
const ride = s("~ ~ rim ~").note(37)${viz.ride}.midichan(6)

const bass = note("<c2 c2 eb2 f2>*8")
  .s("sawtooth").lpf(sine.range(300,900).slow(8)).lpq(6)
  .struct("t(5,8)")${viz.bass}.midichan(7)

const subBass = note("<c1 c1 eb1 f1>*4").s("sine")${viz.subBass}.midichan(8)

const chords = note("<[c4,eb4,g4] [c4,eb4,g4] [ab3,c4,eb4] [f3,ab3,c4]>")
  .s("sawtooth").lpf(perlin.range(400,2000).slow(16))
  .attack(0.01).release(0.3).struct("~ t")${viz.chords}.midichan(9)

const pad = note("<c3 eb3 ab2 f3>")
  .s("sawtooth").lpf(sine.range(200,600).slow(32))
  .attack(1).release(2)${viz.pad}.midichan(10)

const arp = note("<c4 eb4 g4 c5>*8")
  .s("triangle").delay(0.4).delaytime(0.125).delayfeedback(0.4)${viz.arp}.midichan(11)

const fxRoll = s("sd*16").note(38).gain(sine.range(0.1,0.6).slow(4))${viz.fxRoll}.midichan(12)
const marker = note("<96 ~ ~ 97>")${viz.marker}.midichan(13)

const intro     = stack(kick, hats, perc)
const buildUp   = stack(kick, hats, tops, perc, snareClap, bass, subBass)
const breakdown = stack(pad, chords, arp)
const riserUp   = stack(pad, chords, fxRoll, marker)
const drop      = stack(kick, hats, tops, perc, snareClap, ride, bass, subBass, chords, pad, arp)
const outro     = stack(kick, hats, bass, pad)

arrange(
  [8,  intro],
  [8,  buildUp],
  [8,  breakdown],
  [4,  riserUp],
  [16, drop],
  [8,  outro]
)${viz.master}${device ? `.midi('${device}')` : ''}`,

  techno: (device, viz) => `/* 13-Channel DAW Studio Arrangement — Techno */
setcpm(132/4) // 132 bpm

const kick = s("bd*4").note(36)${viz.kick}.midichan(1)
const hats = s("~ hh ~ hh ~ hh ~ hh").note(42).gain(0.7)${viz.hats}.midichan(2)
const tops = s("~ ~ ~ oh").note(46).gain(0.5)${viz.tops}.midichan(3)
const perc = s("perc*8").note(70).gain(0.3)${viz.perc}.midichan(4)
const snareClap = s("~ ~ cp ~").note(39).gain(0.7)${viz.snareClap}.midichan(5)
const ride = s("~ rim ~ ~ ~ ~ ~ rim").note(37).gain(0.4)${viz.ride}.midichan(6)

const bass = note("c2*8")
  .s("sawtooth").lpf(sine.range(200,1200).slow(16))${viz.bass}.midichan(7)

const subBass = note("<c1 ~ c1 ~>").s("sine")${viz.subBass}.midichan(8)

const chords = note("<c5 eb5 c5 ab4>")
  .s("sawtooth").struct("~ ~ t ~").decay(0.15)${viz.chords}.midichan(9)

const pad = note("<c4 eb4>").s("sawtooth").lpf(400).attack(2).release(3)${viz.pad}.midichan(10)

const arp = note("<c5 eb5 g5 bb5>*16")
  .s("triangle").lpf(perlin.range(600,3000).slow(8))${viz.arp}.midichan(11)

const fxRoll = s("sd*16").note(38).gain(sine.range(0.1,0.6).slow(4))${viz.fxRoll}.midichan(12)
const marker = note("<96 ~ ~ 97>")${viz.marker}.midichan(13)

const intro    = stack(kick, hats, perc)
const build    = stack(kick, hats, tops, perc, snareClap, bass, subBass, chords)
const brk      = stack(pad, chords, arp, fxRoll)
const riserUp  = stack(pad, fxRoll, marker)
const mainLoop = stack(kick, hats, tops, perc, snareClap, ride, bass, subBass, chords, pad, arp)
const outro    = stack(kick, hats, bass, pad)

arrange(
  [8,  intro],
  [16, build],
  [8,  brk],
  [4,  riserUp],
  [24, mainLoop],
  [8,  outro]
)${viz.master}${device ? `.midi('${device}')` : ''}`,

  trance: (device, viz) => `/* 13-Channel DAW Studio Arrangement — Trance */
setcpm(138/4) // 138 bpm

const kick = s("bd*4").note(36)${viz.kick}.midichan(1)
const hats = s("~ hh ~ hh ~ hh ~ hh").note(42).gain(0.75)${viz.hats}.midichan(2)
const tops = s("~ oh ~ oh ~ oh ~ oh").note(46)${viz.tops}.midichan(3)
const perc = s("shaker*8").note(70).gain(0.35)${viz.perc}.midichan(4)
const snareClap = s("~ cp ~ cp").note(39).gain(0.95)${viz.snareClap}.midichan(5)
const ride = s("~ ~ rim ~").note(37).gain(0.4)${viz.ride}.midichan(6)

const bass = note("c2*16")
  .s("sawtooth").lpf(sine.range(300,1500).slow(8))${viz.bass}.midichan(7)

const subBass = note("<c1 c1 eb1 f1>*4").s("sine")${viz.subBass}.midichan(8)

const chords = note("<[c4,e4,g4] [c4,e4,g4] [a3,c4,e4] [g3,c4,e4]>")
  .s("supersaw").attack(0.05).release(1.5)${viz.chords}.midichan(9)

const pad = note("<c4 eb4 g4 f4>")
  .s("sawtooth").lpf(800).attack(2).release(3)${viz.pad}.midichan(10)

const arp = note("<c5 e5 g5 c6>*16")
  .s("triangle").lpf(perlin.range(800,4000).slow(8))${viz.arp}.midichan(11)

const fxRoll = s("sd*16").note(38).gain(sine.range(0.1,0.7).slow(4))${viz.fxRoll}.midichan(12)
const marker = note("<96 ~ ~ ~ ~ ~ ~ 97>")${viz.marker}.midichan(13)

const intro     = stack(kick, hats, perc)
const build     = stack(kick, hats, tops, perc, snareClap, bass, subBass)
const breakdown = stack(pad, chords, arp)
const riserUp   = stack(pad, chords, fxRoll, marker)
const drop      = stack(kick, hats, tops, perc, snareClap, ride, bass, subBass, chords, pad, arp)
const outro     = stack(kick, hats, bass, pad)

arrange(
  [8,  intro],
  [16, build],
  [16, breakdown],
  [8,  riserUp],
  [16, drop],
  [8,  outro]
)${viz.master}${device ? `.midi('${device}')` : ''}`,

  dnb: (device, viz) => `/* 13-Channel DAW Studio Arrangement — Drum & Bass */
setcpm(174/4) // 174 bpm

const kick = s("bd ~ ~ ~ ~ bd ~ ~").note(36)${viz.kick}.midichan(1)
const hats = s("hh*16").note(42).gain(0.6)${viz.hats}.midichan(2)
const tops = s("~ ~ ~ oh ~ ~ ~ oh").note(46).gain(0.55)${viz.tops}.midichan(3)
const perc = s("perc*8").note(70).gain(0.15)${viz.perc}.midichan(4)
const snareClap = s("~ ~ sd ~").note(38).gain(1)${viz.snareClap}.midichan(5)

const bass = note("~ ~ ~ c1 ~ ~ ~ ~ ~ ~ ~ c1 ~ ~ ~ ~").s("sine")${viz.bass}.midichan(7)
const subBass = note("~ ~ ~ ~ ~ c0 ~ ~ ~ ~ ~ ~ ~ c0 ~ ~").s("sine")${viz.subBass}.midichan(8)

const chords = note("<[d4,f4,a4] [c4,eb4,g4] [bb3,d4,f4] [g3,bb3,d4]>")
  .s("sawtooth").struct("t ~ ~ ~").decay(0.2)${viz.chords}.midichan(9)

const pad = note("<a3 d4 bb3 c4>").s("sawtooth").attack(1.5).release(2.5)${viz.pad}.midichan(10)
const arp = note("~ a4 ~ c5 ~ d5 ~ f5").s("triangle")${viz.arp}.midichan(11)

const fxRoll = s("sd*16").note(38).gain(0.5)${viz.fxRoll}.midichan(12)
const marker = note("<96 ~ ~ 97>")${viz.marker}.midichan(13)

const intro     = stack(kick, hats, perc)
const build     = stack(kick, hats, tops, perc, snareClap, bass)
const drop1     = stack(kick, hats, tops, perc, snareClap, bass, subBass, chords, arp)
const breakdown = stack(pad, chords, arp)
const riserUp   = stack(pad, fxRoll, marker)
const drop2     = stack(kick, hats, tops, perc, snareClap, bass, subBass, chords, pad, arp)
const outro     = stack(kick, hats, bass, pad)

arrange(
  [8,  intro],
  [8,  build],
  [16, drop1],
  [8,  breakdown],
  [4,  riserUp],
  [16, drop2],
  [8,  outro]
)${viz.master}${device ? `.midi('${device}')` : ''}`,

  dubstep: (device, viz) => `/* 13-Channel DAW Studio Arrangement — Dubstep */
setcpm(140/4) // 140 bpm (half-time)

const kick = s("bd ~ ~ ~").note(36)${viz.kick}.midichan(1)
const hats = s("~ hh hh ~ hh ~ hh ~").note(42).gain(0.5)${viz.hats}.midichan(2)
const perc = s("perc*8").note(70).gain(0.15)${viz.perc}.midichan(4)
const snareClap = s("~ ~ sd ~").note(38).gain(1)${viz.snareClap}.midichan(5)

const bass = note("<c2 c1>*8")
  .s("sawtooth").lpf(sine.range(150,2500).fast(2)).lpq(10)${viz.bass}.midichan(7)
const subBass = note("<c1 ~ c1 ~>").s("sine")${viz.subBass}.midichan(8)

const chords = note("<[d4,f4,a4] [c4,eb4,g4] [bb3,d4,f4] [g3,bb3,d4]>")
  .s("sawtooth").struct("t ~ ~ ~").distort(0.3)${viz.chords}.midichan(9)

const pad = note("<a3 d4 bb3 c4>").s("sawtooth").attack(1.5).release(2.5)${viz.pad}.midichan(10)
const arp = note("~ a4 ~ c5 ~ d5 ~ f5").s("triangle")${viz.arp}.midichan(11)

const fxRoll = s("sd*16").note(38).gain(0.5)${viz.fxRoll}.midichan(12)
const marker = note("<96 ~ ~ 97>")${viz.marker}.midichan(13)

const intro     = stack(kick, hats, perc)
const build     = stack(kick, hats, perc, snareClap, bass)
const drop1     = stack(kick, hats, perc, snareClap, bass, subBass, chords)
const breakdown = stack(pad, chords, arp)
const riserUp   = stack(pad, fxRoll, marker)
const drop2     = stack(kick, hats, perc, snareClap, bass, subBass, chords, pad)
const outro     = stack(kick, hats, bass, pad)

arrange(
  [8,  intro],
  [8,  build],
  [16, drop1],
  [8,  breakdown],
  [4,  riserUp],
  [16, drop2],
  [8,  outro]
)${viz.master}${device ? `.midi('${device}')` : ''}`,

  deepHouse: (device, viz) => `/* 13-Channel DAW Studio Arrangement — Deep House */
setcpm(122/4) // 122 bpm

const kick = s("bd*4").note(36).gain(0.8)${viz.kick}.midichan(1)
const hats = s("hh*8").note(42).swingBy(1/3, 4).gain(0.55)${viz.hats}.midichan(2)
const tops = s("~ ~ ~ oh ~ ~ ~ oh").note(46).gain(0.45)${viz.tops}.midichan(3)
const perc = s("shaker*8").note(70).gain(0.25)${viz.perc}.midichan(4)
const snareClap = s("~ cp ~ cp").note(39).gain(0.8)${viz.snareClap}.midichan(5)
const ride = s("~ rim ~ ~ ~ ~ ~ rim").note(37).gain(0.35)${viz.ride}.midichan(6)

const bass = note("c2 ~ eb2 f2 ~ g2 ~ ~").s("sawtooth").lpf(900)${viz.bass}.midichan(7)
const subBass = note("<c1 c1 eb1 f1>*4").s("sine").gain(0.6)${viz.subBass}.midichan(8)

const chords = note("<[c5,e5,g5,b5] [a4,c5,e5,g5] [d5,f5,a5,c6] [g4,b4,d5,f5]>")
  .s("epiano").struct("~ ~ t ~")${viz.chords}.midichan(9)

const pad = note("<c4 e4 g4 f4>").s("sawtooth").attack(2).release(3).gain(0.28)${viz.pad}.midichan(10)
const arp = note("~ c5 ~ e5 ~ g5 ~ b5").s("triangle").gain(0.4)${viz.arp}.midichan(11)

const fxRoll = s("sd*8").note(38).gain(0.4)${viz.fxRoll}.midichan(12)
const marker = note("<96 ~ ~ 97>")${viz.marker}.midichan(13)

const intro     = stack(kick, hats, perc)
const build     = stack(kick, hats, tops, perc, snareClap, bass)
const groove1   = stack(kick, hats, tops, perc, snareClap, ride, bass, subBass, chords)
const breakdown = stack(pad, chords, arp)
const riserUp   = stack(pad, fxRoll, marker)
const groove2   = stack(kick, hats, tops, perc, snareClap, ride, bass, subBass, chords, pad, arp)
const outro     = stack(kick, hats, bass, pad)

arrange(
  [8,  intro],
  [8,  build],
  [16, groove1],
  [8,  breakdown],
  [4,  riserUp],
  [16, groove2],
  [8,  outro]
)${viz.master}${device ? `.midi('${device}')` : ''}`,
};

// Standard MIDI File (SMF) Type 1 binary generator
const TPQ = 480;
function writeVarLen(value) {
  let buffer = value & 0x7f;
  const bytes = [];
  while ((value >>= 7)) {
    buffer <<= 8;
    buffer |= (value & 0x7f) | 0x80;
  }
  while (true) {
    bytes.push(buffer & 0xff);
    if (buffer & 0x80) buffer >>= 8;
    else break;
  }
  return bytes;
}
function u16(n) {
  return [(n >> 8) & 0xff, n & 0xff];
}
function u32(n) {
  return [(n >> 24) & 0xff, (n >> 16) & 0xff, (n >> 8) & 0xff, n & 0xff];
}
function textBytes(s) {
  return Array.from(s).map((c) => c.charCodeAt(0));
}

function buildTrackChunk(name, channel, events, isConductor, bpm) {
  const raw = [];
  if (isConductor) {
    raw.push({ tick: 0, bytes: [0xff, 0x58, 0x04, 0x04, 0x02, 0x18, 0x08] });
    const usPerQ = Math.round(60000000 / bpm);
    raw.push({ tick: 0, bytes: [0xff, 0x51, 0x03, (usPerQ >> 16) & 0xff, (usPerQ >> 8) & 0xff, usPerQ & 0xff] });
  }
  raw.push({ tick: 0, bytes: [0xff, 0x03, name.length, ...textBytes(name)] });
  events.forEach((e) => {
    raw.push({ tick: e.tick, bytes: [0x90 | (channel - 1), e.note, e.vel] });
    raw.push({ tick: e.tick + e.dur, bytes: [0x80 | (channel - 1), e.note, 0] });
  });
  raw.sort((a, b) => a.tick - b.tick);
  let bytes = [],
    lastTick = 0;
  raw.forEach((ev) => {
    const delta = Math.max(0, Math.round(ev.tick - lastTick));
    bytes = bytes.concat(writeVarLen(delta), ev.bytes);
    lastTick = ev.tick;
  });
  bytes = bytes.concat(writeVarLen(0), [0xff, 0x2f, 0x00]);
  return [...textBytes('MTrk'), ...u32(bytes.length), ...bytes];
}

export default function DawArrangerTab({ context }) {
  const [selectedGenre, setSelectedGenre] = useState('progHouse');
  const [bpm, setBpm] = useState(130);
  const [midiDevices, setMidiDevices] = useState([]);
  const [selectedDevice, setSelectedDevice] = useState('IAC Driver');
  const [liveMidiActive, setLiveMidiActive] = useState(true);
  const [statusLog, setStatusLog] = useState('');
  const [copied, setCopied] = useState(false);
  const [injected, setInjected] = useState(false);

  // Visualizers state
  const [vizPunchcard, setVizPunchcard] = useState(true);
  const [vizPianoroll, setVizPianoroll] = useState(true);
  const [vizScope, setVizScope] = useState(true);
  const [vizSpectrum, setVizSpectrum] = useState(false);
  const [vizColor, setVizColor] = useState(true);

  // Initialize WebMIDI & list devices
  useEffect(() => {
    enableWebMidi()
      .then((wm) => {
        const names = wm.outputs.map((o) => o.name);
        setMidiDevices(names);
        if (names.length > 0) {
          const iac = names.find((n) => n.includes('IAC') || n.includes('Loop') || n.includes('virtual'));
          setSelectedDevice(iac || names[0]);
        }
      })
      .catch((err) => {
        console.warn('[DawArranger] WebMIDI init:', err.message);
      });
  }, []);

  const handleGenreChange = (genreKey) => {
    setSelectedGenre(genreKey);
    const g = GENRES[genreKey];
    if (g) setBpm(g.bpm);
  };

  const getVisualizerStrings = () => {
    const out = {};
    CHANNELS.forEach((ch) => {
      if (ch.type === 'punchcard') {
        if (!vizPunchcard) out[ch.id] = '';
        else if (vizColor)
          out[ch.id] = `._punchcard({active:'${ch.color}', background:'#111111'})`;
        else out[ch.id] = '._punchcard()';
      } else if (ch.type === 'pianoroll') {
        if (!vizPianoroll) out[ch.id] = '';
        else if (vizColor)
          out[ch.id] = `._pianoroll({cycles:2, active:'${ch.color}', background:'#111111'})`;
        else out[ch.id] = '._pianoroll({cycles:2})';
      }
    });

    const master = [];
    if (vizScope) master.push(vizColor ? `._scope({color:'#00FFC3'})` : '._scope()');
    if (vizSpectrum) master.push('._spectrum()');
    out.master = master.join('');
    return out;
  };

  const getGeneratedCode = () => {
    const fn = STRUDEL_PATTERNS[selectedGenre] || STRUDEL_PATTERNS.progHouse;
    const viz = getVisualizerStrings();
    const dev = liveMidiActive ? selectedDevice : '';
    return fn(dev, viz);
  };

  const testTriggerChannel = (chNum, note) => {
    if (!WebMidi.enabled) {
      enableWebMidi()
        .then(() => testTriggerChannel(chNum, note))
        .catch(() => alert('WebMIDI not supported in this browser.'));
      return;
    }
    const output = WebMidi.outputs.find((o) => o.name === selectedDevice) || WebMidi.outputs[0];
    if (!output) {
      setStatusLog(`⚠️ No MIDI output selected.`);
      return;
    }
    output.sendNoteOn(note, { channels: [chNum], velocity: 0.8 });
    setTimeout(() => {
      output.sendNoteOff(note, { channels: [chNum] });
    }, 180);
    setStatusLog(`⚡ Triggered Note ${note} on Channel ${chNum} (${output.name})`);
  };

  const handlePanic = () => {
    if (!WebMidi.enabled) return;
    WebMidi.outputs.forEach((out) => {
      out.sendAllNotesOff();
      out.sendAllSoundOff();
    });
    setStatusLog(`🛑 Panic: Sent All Notes Off & All Sound Off to all MIDI devices.`);
  };

  const handleInjectCode = () => {
    const code = getGeneratedCode();
    if (context?.editorRef?.current) {
      context.editorRef.current.setCode(code);
      context.editorRef.current.evaluate();
      setInjected(true);
      setTimeout(() => setInjected(false), 2000);
      setStatusLog(`🚀 Injected multi-channel arrangement into REPL editor!`);
    } else {
      navigator.clipboard.writeText(code);
      setStatusLog(`📋 Code copied to clipboard (editor ref unavailable)`);
    }
  };

  const handleCopyCode = () => {
    const code = getGeneratedCode();
    navigator.clipboard.writeText(code);
    setCopied(true);
    setTimeout(() => setCopied(false), 1500);
  };

  const handleDownloadMidi = () => {
    const genre = GENRES[selectedGenre];
    chordCounter = 0;
    padCounter = 0;

    const trackEvents = {};
    CHANNELS.forEach((k) => (trackEvents[k.id] = []));

    let barCursor = 0;
    genre.sections.forEach((section) => {
      for (let b = 0; b < section.bars; b++) {
        const barStartTick = barCursor * 4 * TPQ;
        section.active.forEach((roleKey) => {
          if (roleKey === 'marker') {
            if (b === 0) trackEvents.marker.push({ tick: barStartTick, note: 96, dur: TPQ * 0.25, vel: 110 });
            if (b === section.bars - 1)
              trackEvents.marker.push({ tick: barStartTick, note: 97, dur: TPQ * 0.25, vel: 110 });
            return;
          }
          if (genre.roles[roleKey]) {
            const hits = genre.roles[roleKey]();
            hits.forEach((h) => {
              trackEvents[roleKey].push({
                tick: barStartTick + h.beat * TPQ,
                note: h.note,
                dur: h.dur * TPQ,
                vel: h.vel,
              });
            });
          }
        });
        barCursor++;
      }
    });

    const chunks = [];
    const activeTracks = CHANNELS.filter((c) => (trackEvents[c.id] || []).length > 0);
    const totalTracks = activeTracks.length + 1; // + Conductor
    const header = [...textBytes('MThd'), ...u32(6), ...u16(1), ...u16(totalTracks), ...u16(TPQ)];
    chunks.push(buildTrackChunk('Conductor', 1, [], true, bpm));

    let totalNotes = 0;
    activeTracks.forEach((c) => {
      const evs = trackEvents[c.id];
      totalNotes += evs.length;
      chunks.push(buildTrackChunk(c.name, c.ch, evs, false, bpm));
    });

    const fileBytes = new Uint8Array(header.length + chunks.reduce((s, c) => s + c.length, 0));
    let offset = 0;
    header.forEach((b) => (fileBytes[offset++] = b));
    chunks.forEach((chunk) => chunk.forEach((b) => (fileBytes[offset++] = b)));

    const blob = new Blob([fileBytes], { type: 'audio/midi' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${genre.label.replace(/\s+/g, '-').toLowerCase()}-${bpm}bpm-arrangement.mid`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);

    setStatusLog(`💾 Downloaded ${a.download} (${totalNotes} notes across ${activeTracks.length} tracks).`);
  };

  const activeGenre = GENRES[selectedGenre] || GENRES.progHouse;
  const totalBars = activeGenre.sections.reduce((sum, s) => sum + s.bars, 0);

  return (
    <div className="text-foreground p-4 space-y-4 max-w-full font-sans text-xs">
      {/* Header */}
      <div className="flex items-center justify-between border-b border-muted pb-2">
        <div>
          <h2 className="text-sm font-bold text-foreground">DAW Studio & Arranger Bridge</h2>
          <p className="text-muted text-[11px]">13-Channel Multi-Track Bitwig / DAW Live MIDI & Arranger Engine</p>
        </div>
        <button
          onClick={handlePanic}
          title="Panic: Send All Notes Off"
          className="bg-red-900/40 hover:bg-red-800/60 border border-red-700 text-red-300 font-semibold px-2 py-1 rounded text-[11px] transition-colors"
        >
          🛑 Panic
        </button>
      </div>

      {/* Live MIDI Device Routing */}
      <div className="bg-background/50 border border-muted p-3 rounded-md space-y-2">
        <div className="flex items-center justify-between">
          <span className="font-semibold text-foreground text-[11px] uppercase tracking-wider">
            Live MIDI Output Routing
          </span>
          <label className="flex items-center space-x-1 cursor-pointer">
            <input
              type="checkbox"
              checked={liveMidiActive}
              onChange={(e) => setLiveMidiActive(e.target.checked)}
              className="rounded bg-background"
            />
            <span className="text-muted text-[11px]">Enable Live DAW Output</span>
          </label>
        </div>

        <div className="flex gap-2 items-center">
          <select
            value={selectedDevice}
            onChange={(e) => setSelectedDevice(e.target.value)}
            disabled={!liveMidiActive}
            className={cx(
              'bg-lineHighlight border border-muted text-foreground text-xs p-1.5 rounded grow',
              !liveMidiActive && 'opacity-40',
            )}
          >
            {midiDevices.length === 0 ? (
              <option value="IAC Driver">IAC Driver (macOS default)</option>
            ) : (
              midiDevices.map((d) => (
                <option key={d} value={d}>
                  {d}
                </option>
              ))
            )}
          </select>
          <button
            onClick={() => {
              enableWebMidi()
                .then((wm) => {
                  setMidiDevices(wm.outputs.map((o) => o.name));
                  setStatusLog(`🔄 Refreshed MIDI Devices: Found ${wm.outputs.length} outputs.`);
                })
                .catch((e) => setStatusLog(`Error: ${e.message}`));
            }}
            className="bg-lineHighlight border border-muted px-2 py-1 rounded hover:bg-background text-[11px]"
          >
            🔄 Refresh
          </button>
        </div>
      </div>

      {/* Genre & Tempo Controls */}
      <div className="grid grid-cols-2 gap-3">
        <div>
          <label className="block text-muted text-[11px] mb-1">Electronic Genre</label>
          <select
            value={selectedGenre}
            onChange={(e) => handleGenreChange(e.target.value)}
            className="bg-lineHighlight border border-muted text-foreground text-xs p-1.5 rounded w-full"
          >
            {Object.keys(GENRES).map((k) => (
              <option key={k} value={k}>
                {GENRES[k].label}
              </option>
            ))}
          </select>
        </div>
        <div>
          <label className="block text-muted text-[11px] mb-1">Tempo (BPM)</label>
          <input
            type="number"
            value={bpm}
            onChange={(e) => setBpm(Math.max(20, Math.min(300, parseInt(e.target.value) || 120)))}
            className="bg-lineHighlight border border-muted text-foreground text-xs p-1.5 rounded w-full"
          />
        </div>
      </div>

      {/* Structure Display */}
      <div className="bg-background/40 border border-muted p-2 rounded text-[11px]">
        <span className="text-muted font-medium">Arrangement ({totalBars} bars): </span>
        <span className="text-foreground">
          {activeGenre.sections.map((s) => `${s.name} (${s.bars}b)`).join(' → ')}
        </span>
      </div>

      {/* 13-Channel Multi-Track Matrix */}
      <div>
        <div className="flex justify-between items-center mb-1">
          <span className="font-semibold text-foreground text-[11px] uppercase tracking-wider">
            13-Channel DAW Track Matrix
          </span>
          <span className="text-[10px] text-muted">Fixed Bitwig Layout</span>
        </div>
        <div className="border border-muted rounded overflow-hidden max-h-48 overflow-y-auto">
          <table className="w-full text-[11px] text-left border-collapse">
            <thead className="bg-lineHighlight border-b border-muted text-muted font-semibold">
              <tr>
                <th className="p-1 pl-2">CH</th>
                <th className="p-1">Track Role</th>
                <th className="p-1">Type</th>
                <th className="p-1 pr-2 text-right">Test Trigger</th>
              </tr>
            </thead>
            <tbody>
              {CHANNELS.map((ch) => (
                <tr key={ch.id} className="border-b border-muted/50 hover:bg-background/50">
                  <td className="p-1 pl-2 font-mono font-bold" style={{ color: ch.color }}>
                    CH {ch.ch}
                  </td>
                  <td className="p-1 text-foreground font-medium">{ch.name}</td>
                  <td className="p-1 text-muted">{ch.type}</td>
                  <td className="p-1 pr-2 text-right">
                    <button
                      onClick={() => testTriggerChannel(ch.ch, ch.defaultNote)}
                      className="bg-lineHighlight hover:bg-muted border border-muted px-1.5 py-0.5 rounded text-[10px] text-foreground font-mono"
                    >
                      ▶ #{ch.defaultNote}
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Visualizers Toggles */}
      <div className="bg-background/50 border border-muted p-2.5 rounded space-y-1.5">
        <span className="block font-semibold text-foreground text-[11px] uppercase tracking-wider">
          Track Visualizers
        </span>
        <div className="grid grid-cols-3 gap-2 text-[11px] text-muted">
          <label className="flex items-center space-x-1 cursor-pointer">
            <input
              type="checkbox"
              checked={vizPunchcard}
              onChange={(e) => setVizPunchcard(e.target.checked)}
              className="rounded bg-background"
            />
            <span>Punchcard</span>
          </label>
          <label className="flex items-center space-x-1 cursor-pointer">
            <input
              type="checkbox"
              checked={vizPianoroll}
              onChange={(e) => setVizPianoroll(e.target.checked)}
              className="rounded bg-background"
            />
            <span>Pianoroll</span>
          </label>
          <label className="flex items-center space-x-1 cursor-pointer">
            <input
              type="checkbox"
              checked={vizScope}
              onChange={(e) => setVizScope(e.target.checked)}
              className="rounded bg-background"
            />
            <span>Oscilloscope</span>
          </label>
          <label className="flex items-center space-x-1 cursor-pointer">
            <input
              type="checkbox"
              checked={vizSpectrum}
              onChange={(e) => setVizSpectrum(e.target.checked)}
              className="rounded bg-background"
            />
            <span>Spectrum</span>
          </label>
          <label className="flex items-center space-x-1 cursor-pointer">
            <input
              type="checkbox"
              checked={vizColor}
              onChange={(e) => setVizColor(e.target.checked)}
              className="rounded bg-background"
            />
            <span>Track Colors</span>
          </label>
        </div>
      </div>

      {/* Action Buttons */}
      <div className="grid grid-cols-2 gap-2 pt-1">
        <button
          onClick={handleInjectCode}
          className="bg-blue-600 hover:bg-blue-500 text-white font-semibold py-2 px-3 rounded shadow flex items-center justify-center space-x-1.5 transition-all text-xs"
        >
          <span>🚀</span>
          <span>{injected ? 'Injected!' : 'Inject to REPL Editor'}</span>
        </button>
        <button
          onClick={handleDownloadMidi}
          className="bg-lineHighlight hover:bg-background border border-muted text-foreground font-semibold py-2 px-3 rounded shadow flex items-center justify-center space-x-1.5 transition-all text-xs"
        >
          <span>💾</span>
          <span>Download .MID to Bitwig</span>
        </button>
      </div>

      {/* Status Log */}
      {statusLog && (
        <div className="bg-lineHighlight border border-muted p-2 rounded font-mono text-[10px] text-green-400 break-words">
          {statusLog}
        </div>
      )}

      {/* Code Preview Accordion */}
      <details className="border border-muted rounded bg-background/50">
        <summary className="p-2 cursor-pointer font-semibold text-[11px] text-muted hover:text-foreground flex justify-between items-center">
          <span>Generated Strudel Code Preview</span>
          <button
            onClick={(e) => {
              e.preventDefault();
              handleCopyCode();
            }}
            className="text-[10px] bg-lineHighlight px-2 py-0.5 rounded border border-muted"
          >
            {copied ? 'Copied!' : 'Copy'}
          </button>
        </summary>
        <pre className="p-2 text-[10px] font-mono text-muted/90 max-h-40 overflow-y-auto whitespace-pre-wrap border-t border-muted">
          {getGeneratedCode()}
        </pre>
      </details>
    </div>
  );
}
