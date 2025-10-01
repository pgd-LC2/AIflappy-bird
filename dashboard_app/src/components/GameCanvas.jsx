import { useEffect, useRef, useState } from 'react'
import { Play, Pause, RotateCcw, Zap } from 'lucide-react'

export default function GameCanvas({ gameData, onStart, onPause, onReset, isPlaying, speed, onSpeedChange }) {
  const canvasRef = useRef(null)
  
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || !gameData) return
    
    const ctx = canvas.getContext('2d')
    const width = 400
    const height = 600
    
    ctx.clearRect(0, 0, width, height)
    
    const gradient = ctx.createLinearGradient(0, 0, 0, height)
    gradient.addColorStop(0, '#87CEEB')
    gradient.addColorStop(1, '#E0F6FF')
    ctx.fillStyle = gradient
    ctx.fillRect(0, 0, width, height)
    
    if (gameData.pipes) {
      gameData.pipes.forEach(pipe => {
        ctx.fillStyle = '#22c55e'
        ctx.shadowColor = 'rgba(0, 0, 0, 0.1)'
        ctx.shadowBlur = 8
        ctx.fillRect(pipe.x, 0, 80, pipe.gap_y)
        ctx.fillRect(pipe.x, pipe.gap_y + 200, 80, height - pipe.gap_y - 200)
        ctx.shadowBlur = 0
      })
    }
    
    if (gameData.bird_y !== undefined) {
      const birdY = gameData.bird_y
      const birdX = 100
      
      ctx.beginPath()
      ctx.arc(birdX + 10, birdY + 10, 10, 0, Math.PI * 2)
      ctx.fillStyle = '#fbbf24'
      ctx.fill()
      ctx.strokeStyle = '#f59e0b'
      ctx.lineWidth = 2
      ctx.stroke()
      
      if (gameData.action === 1) {
        ctx.beginPath()
        ctx.moveTo(birdX + 10, birdY - 5)
        ctx.lineTo(birdX + 5, birdY - 15)
        ctx.lineTo(birdX + 15, birdY - 15)
        ctx.closePath()
        ctx.fillStyle = '#ef4444'
        ctx.fill()
      }
    }
    
    ctx.font = 'bold 24px Inter'
    ctx.fillStyle = 'white'
    ctx.strokeStyle = 'rgba(0, 0, 0, 0.3)'
    ctx.lineWidth = 4
    ctx.strokeText(`分数: ${gameData.score || 0}`, 20, 40)
    ctx.fillText(`分数: ${gameData.score || 0}`, 20, 40)
    
    if (gameData.game_over) {
      ctx.fillStyle = 'rgba(0, 0, 0, 0.7)'
      ctx.fillRect(0, 0, width, height)
      
      ctx.font = 'bold 48px Inter'
      ctx.fillStyle = 'white'
      ctx.textAlign = 'center'
      ctx.strokeText('游戏结束', width / 2, height / 2)
      ctx.fillText('游戏结束', width / 2, height / 2)
      
      ctx.font = '24px Inter'
      ctx.strokeText(`最终得分: ${gameData.score}`, width / 2, height / 2 + 50)
      ctx.fillText(`最终得分: ${gameData.score}`, width / 2, height / 2 + 50)
    }
  }, [gameData])
  
  return (
    <div className="glass rounded-2xl p-6">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-lg font-semibold text-gray-900">AI实时演示</h2>
        <div className="flex items-center gap-2">
          <button
            onClick={onSpeedChange}
            className="px-3 py-1.5 rounded-lg bg-gray-100 hover:bg-gray-200 text-gray-700 text-sm font-medium transition-colors flex items-center gap-1.5"
          >
            <Zap className="w-4 h-4" />
            {speed}x
          </button>
          <button
            onClick={onPause}
            disabled={!isPlaying}
            className="px-3 py-1.5 rounded-lg bg-gray-100 hover:bg-gray-200 disabled:opacity-50 disabled:cursor-not-allowed text-gray-700 transition-colors"
          >
            {isPlaying ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
          </button>
          <button
            onClick={onReset}
            className="px-3 py-1.5 rounded-lg bg-gray-100 hover:bg-gray-200 text-gray-700 transition-colors"
          >
            <RotateCcw className="w-4 h-4" />
          </button>
          <button
            onClick={onStart}
            className="px-4 py-1.5 rounded-lg bg-gradient-to-r from-blue-500 to-indigo-600 hover:from-blue-600 hover:to-indigo-700 text-white font-medium transition-all shadow-md hover:shadow-lg"
          >
            开始新游戏
          </button>
        </div>
      </div>
      
      <div className="relative rounded-xl overflow-hidden shadow-2xl">
        <canvas
          ref={canvasRef}
          width={400}
          height={600}
          className="w-full"
        />
      </div>
    </div>
  )
}
