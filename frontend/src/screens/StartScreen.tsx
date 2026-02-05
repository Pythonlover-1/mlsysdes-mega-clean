import UploadArea from '../components/start/UploadArea';
import ParametersPanel from '../components/start/ParametersPanel';
import StartHistoryList from '../components/start/StartHistoryList';

export default function StartScreen() {
  return (
    <div className="min-h-screen bg-black">
      <div className="max-w-4xl mx-auto p-6 space-y-8">
        {/* Header */}
        <div className="text-center space-y-4 py-8">
          <h1 className="text-5xl md:text-6xl font-black text-white tracking-tight">
            BPMN Editor
          </h1>
          <p className="text-xl text-gray-400 max-w-2xl mx-auto">
            Преобразуйте диаграмму в интерактивный граф и редактируйте его
          </p>
        </div>

        {/* Upload Area */}
        <UploadArea />

        {/* Parameters (placeholder) */}
        <ParametersPanel />

        {/* History */}
        <StartHistoryList />
      </div>
    </div>
  );
}
