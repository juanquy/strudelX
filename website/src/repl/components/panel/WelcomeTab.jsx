import { useSettings } from '@src/settings.mjs';

const { BASE_URL } = import.meta.env;
const baseNoTrailing = BASE_URL.endsWith('/') ? BASE_URL.slice(0, -1) : BASE_URL;

export function WelcomeTab({ context }) {
  const { fontFamily } = useSettings();
  return (
    <div className="prose dark:prose-invert min-w-full py-4 font-sans px-4 text-sm" style={{ fontFamily }}>
      <h3>welcome</h3>
      <p>
        You have found <span className="underline">strudel</span>, a new live coding platform to write dynamic music
        pieces in the browser! It is free and open-source and made for beginners and experts alike. To get started:
        <br />
        <br />
        <span className="underline">1. hit play</span> - <span className="underline">2. change something</span> -{' '}
        <span className="underline">3. hit update</span>
        {/* <br />
        If you don't like what you hear, try <span className="underline">shuffle</span>! */}
      </p>
      <p>
        To get started, check out the{' '}
        <a href={`${baseNoTrailing}/workshop/getting-started/`} target="_blank">
          interactive tutorial
        </a>
        . Also feel free to join the{' '}
        <a href="https://discord.com/invite/HGEdXmRkzT" target="_blank">
          discord channel
        </a>{' '}
        to ask any questions, give feedback or just say hello.
      </p>

      <div className="my-6 p-4 rounded-lg bg-lineHighlight border border-muted">
        <h4 className="text-foreground font-bold mb-3 text-base flex items-center gap-2">
          <span>⚡ Live Performance Keyboard Shortcuts</span>
        </h4>
        <div className="grid grid-cols-1 gap-2 text-xs font-mono">
          <div className="flex justify-between py-1 border-b border-muted">
            <span className="text-blue-400 font-bold">Ctrl + Enter / Cmd + Enter</span>
            <span>Evaluate & Play Pattern</span>
          </div>
          <div className="flex justify-between py-1 border-b border-muted">
            <span className="text-red-400 font-bold">Ctrl + . / Cmd + .</span>
            <span>Hush / Stop All Audio</span>
          </div>
          <div className="flex justify-between py-1 border-b border-muted">
            <span className="text-yellow-400 font-bold">Shift + Enter</span>
            <span>Evaluate Current Line</span>
          </div>
          <div className="flex justify-between py-1 border-b border-muted">
            <span className="text-purple-400 font-bold">Ctrl + Shift + Enter</span>
            <span>Evaluate Code Block</span>
          </div>
          <div className="flex justify-between py-1 border-b border-muted">
            <span className="text-green-400 font-bold">Alt + Up / Alt + Down</span>
            <span>Increment / Decrement Number</span>
          </div>
          <div className="flex justify-between py-1">
            <span className="text-cyan-400 font-bold">Ctrl + M / Cmd + M</span>
            <span>Toggle Menu Drawer Panel</span>
          </div>
        </div>
      </div>
      <h3>about</h3>
      <p>
        strudel is a JavaScript version of{' '}
        <a href="https://tidalcycles.org/" target="_blank">
          tidalcycles
        </a>
        , which is a popular live coding language for music, written in Haskell. Strudel is free/open source software,
        with copyright owned by its [contributors](https://codeberg.org/uzu/strudel/activity/contributors). You can
        redistribute and/or modify it under the terms of the{' '}
        <a href="https://codeberg.org/uzu/strudel/src/branch/main/LICENSE" target="_blank">
          GNU Affero General Public License
        </a>
        . You can find the source code at{' '}
        <a href="https://codeberg.org/uzu/strudel" target="_blank">
          codeberg
        </a>
        . You can also find <a href="https://github.com/felixroos/dough-samples/blob/main/README.md">licensing info</a>{' '}
        for the default sound banks there. Please consider to{' '}
        <a href="https://opencollective.com/tidalcycles" target="_blank">
          support this project
        </a>{' '}
        to ensure ongoing development 💖
      </p>
    </div>
  );
}
