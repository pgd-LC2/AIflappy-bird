import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Area, AreaChart } from 'recharts'

export default function TrainingChart({ data, title, dataKey, color = "#3b82f6" }) {
  if (!data || data.length === 0) {
    return (
      <div className="glass rounded-2xl p-6">
        <h2 className="text-lg font-semibold text-gray-900 mb-4">{title}</h2>
        <div className="h-64 flex items-center justify-center text-gray-400">
          暂无训练数据
        </div>
      </div>
    )
  }
  
  const recentData = data.slice(-100)
  
  return (
    <div className="glass rounded-2xl p-6">
      <h2 className="text-lg font-semibold text-gray-900 mb-4">{title}</h2>
      <ResponsiveContainer width="100%" height={280}>
        <AreaChart data={recentData} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
          <defs>
            <linearGradient id={`gradient-${dataKey}`} x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor={color} stopOpacity={0.3}/>
              <stop offset="95%" stopColor={color} stopOpacity={0}/>
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
          <XAxis 
            dataKey="episode" 
            stroke="#9ca3af"
            tick={{ fill: '#6b7280', fontSize: 12 }}
            tickLine={{ stroke: '#e5e7eb' }}
          />
          <YAxis 
            stroke="#9ca3af"
            tick={{ fill: '#6b7280', fontSize: 12 }}
            tickLine={{ stroke: '#e5e7eb' }}
          />
          <Tooltip 
            contentStyle={{
              backgroundColor: 'rgba(255, 255, 255, 0.95)',
              border: '1px solid #e5e7eb',
              borderRadius: '12px',
              boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)',
              padding: '12px'
            }}
            labelStyle={{ color: '#111827', fontWeight: 600, marginBottom: '4px' }}
            itemStyle={{ color: '#6b7280', fontSize: '13px' }}
          />
          <Area 
            type="monotone" 
            dataKey={dataKey} 
            stroke={color}
            strokeWidth={2.5}
            fill={`url(#gradient-${dataKey})`}
            animationDuration={300}
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  )
}
