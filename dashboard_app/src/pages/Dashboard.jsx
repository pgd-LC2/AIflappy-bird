import { useEffect, useState, useCallback } from 'react'
import { Brain, Trophy, TrendingUp, Clock } from 'lucide-react'
import StatCard from '../components/StatCard'
import GameCanvas from '../components/GameCanvas'
import TrainingChart from '../components/TrainingChart'
import { api } from '../lib/api'
import { formatNumber } from '../lib/utils'

export default function Dashboard() {
  const [gameData, setGameData] = useState(null)
  const [trainingData, setTrainingData] = useState([])
  const [isPlaying, setIsPlaying] = useState(false)
  const [speed, setSpeed] = useState(1)
  const [stats, setStats] = useState({
    currentScore: 0,
    highScore: 0,
    totalEpisodes: 0,
    avgScore: 0
  })
  
  const loadTrainingData = useCallback(async () => {
    try {
      const data = await api.getTrainingLog()
      if (data.length > 0) {
        setTrainingData(data)
        
        const scores = data.map(d => d.score)
        const highScore = Math.max(...scores)
        const recentScores = scores.slice(-100)
        const avgScore = recentScores.reduce((a, b) => a + b, 0) / recentScores.length
        
        setStats(prev => ({
          ...prev,
          totalEpisodes: data.length,
          highScore,
          avgScore: avgScore.toFixed(1)
        }))
      }
    } catch (error) {
      console.error('Failed to load training data:', error)
    }
  }, [])
  
  useEffect(() => {
    loadTrainingData()
    const interval = setInterval(loadTrainingData, 5000)
    return () => clearInterval(interval)
  }, [loadTrainingData])
  
  useEffect(() => {
    if (!isPlaying) return
    
    const interval = setInterval(async () => {
      try {
        const data = await api.stepGame()
        setGameData(data)
        setStats(prev => ({
          ...prev,
          currentScore: data.score
        }))
        
        if (data.game_over) {
          setIsPlaying(false)
        }
      } catch (error) {
        console.error('Game step failed:', error)
        setIsPlaying(false)
      }
    }, 100 / speed)
    
    return () => clearInterval(interval)
  }, [isPlaying, speed])
  
  const handleStart = async () => {
    try {
      const data = await api.startGame()
      setGameData(data)
      setIsPlaying(true)
    } catch (error) {
      console.error('Failed to start game:', error)
    }
  }
  
  const handlePause = () => {
    setIsPlaying(!isPlaying)
  }
  
  const handleReset = async () => {
    setIsPlaying(false)
    await handleStart()
  }
  
  const handleSpeedChange = () => {
    setSpeed(prev => {
      if (prev === 1) return 2
      if (prev === 2) return 4
      return 1
    })
  }
  
  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 via-blue-50 to-indigo-50">
      <header className="glass border-b border-gray-200/50 sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-blue-500 to-indigo-600 flex items-center justify-center">
                <Brain className="w-6 h-6 text-white" />
              </div>
              <div>
                <h1 className="text-xl font-bold text-gray-900">Flappy Bird AI 训练监控台</h1>
                <p className="text-sm text-gray-500">基于PPO强化学习算法</p>
              </div>
            </div>
            <div className="flex items-center gap-2">
              <div className="px-3 py-1.5 rounded-lg bg-green-100 text-green-700 text-sm font-medium flex items-center gap-2">
                <span className="w-2 h-2 rounded-full bg-green-500 animate-pulse"></span>
                训练中
              </div>
            </div>
          </div>
        </div>
      </header>
      
      <main className="max-w-7xl mx-auto px-6 py-8">
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
          <StatCard
            title="当前得分"
            value={stats.currentScore}
            subtitle="本局游戏"
            icon={Trophy}
          />
          <StatCard
            title="历史最高"
            value={formatNumber(stats.highScore)}
            subtitle="所有训练记录"
            icon={TrendingUp}
            trend={stats.highScore > 100 ? 100 : 0}
          />
          <StatCard
            title="平均得分"
            value={stats.avgScore}
            subtitle="最近100局"
            icon={Brain}
          />
          <StatCard
            title="训练轮数"
            value={formatNumber(stats.totalEpisodes)}
            subtitle="累计训练"
            icon={Clock}
          />
        </div>
        
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-8">
          <div className="lg:col-span-1">
            <GameCanvas
              gameData={gameData}
              onStart={handleStart}
              onPause={handlePause}
              onReset={handleReset}
              isPlaying={isPlaying}
              speed={speed}
              onSpeedChange={handleSpeedChange}
            />
          </div>
          
          <div className="lg:col-span-2 space-y-6">
            <TrainingChart
              data={trainingData}
              title="得分趋势"
              dataKey="score"
              color="#3b82f6"
            />
            <TrainingChart
              data={trainingData}
              title="奖励趋势"
              dataKey="reward"
              color="#8b5cf6"
            />
          </div>
        </div>
        
        <div className="glass rounded-2xl p-6">
          <h2 className="text-lg font-semibold text-gray-900 mb-4">训练进度</h2>
          <div className="space-y-4">
            {trainingData.slice(-10).reverse().map((episode, idx) => (
              <div key={idx} className="flex items-center justify-between py-3 border-b border-gray-100 last:border-0">
                <div className="flex items-center gap-4">
                  <span className="text-sm font-medium text-gray-500 w-24">
                    Episode {episode.episode}
                  </span>
                  <div className="flex items-center gap-2">
                    <span className="text-sm font-semibold text-gray-900">
                      得分: {episode.score}
                    </span>
                    <span className="text-sm text-gray-500">
                      奖励: {episode.reward.toFixed(1)}
                    </span>
                    <span className="text-sm text-gray-500">
                      步数: {episode.length}
                    </span>
                  </div>
                </div>
                {episode.score >= 100 && (
                  <span className="px-2 py-1 rounded-full bg-yellow-100 text-yellow-700 text-xs font-medium">
                    🏆 高分
                  </span>
                )}
              </div>
            ))}
          </div>
        </div>
      </main>
      
      <footer className="mt-12 py-6 border-t border-gray-200">
        <div className="max-w-7xl mx-auto px-6 text-center text-sm text-gray-500">
          <p>使用PPO算法训练的Flappy Bird AI · 持续学习中</p>
        </div>
      </footer>
    </div>
  )
}
