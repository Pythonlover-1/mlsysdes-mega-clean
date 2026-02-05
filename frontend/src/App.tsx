import { useAppStore } from './store';
import StartScreen from './screens/StartScreen';
import EditorScreen from './screens/EditorScreen';

export default function App() {
  const screen = useAppStore((s) => s.screen);

  if (screen === 'editor') {
    return <EditorScreen />;
  }

  return <StartScreen />;
}
