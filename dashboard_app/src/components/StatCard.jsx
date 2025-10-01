import { cn } from '../lib/utils'

export default function StatCard({ title, value, subtitle, icon: Icon, trend, className }) {
  return (
    <div className={cn(
      "glass rounded-2xl p-6 transition-all duration-300 hover:shadow-lg",
      className
    )}>
      <div className="flex items-start justify-between mb-3">
        <div className="flex-1">
          <p className="text-sm font-medium text-gray-500 mb-1">{title}</p>
          <h3 className="text-3xl font-bold text-gray-900 tracking-tight">{value}</h3>
        </div>
        {Icon && (
          <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-blue-500 to-indigo-600 flex items-center justify-center">
            <Icon className="w-6 h-6 text-white" />
          </div>
        )}
      </div>
      {subtitle && (
        <p className="text-sm text-gray-600 flex items-center gap-2">
          {trend && (
            <span className={cn(
              "inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium",
              trend > 0 ? "bg-green-100 text-green-800" : "bg-gray-100 text-gray-600"
            )}>
              {trend > 0 ? '↑' : '→'} {Math.abs(trend)}%
            </span>
          )}
          {subtitle}
        </p>
      )}
    </div>
  )
}
