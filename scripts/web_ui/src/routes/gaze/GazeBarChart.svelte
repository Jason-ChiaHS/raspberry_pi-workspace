<script lang="ts">
	import { BarChart } from 'layerchart';

	interface GazeBarChartProps {
		data: { dt_time: string; gaze_people: number; non_gaze_people: number }[];
	}
	let { data }: GazeBarChartProps = $props();
	let formatted_data = $derived(
		data.map((gaze_people_data) => {
			let dt_time = new Date(gaze_people_data.dt_time);
			return { ...gaze_people_data, dt_time: dt_time.toTimeString().slice(0, 8) };
		})
	);
</script>

<BarChart
	data={formatted_data}
	x="dt_time"
	series={[
		{
			key: 'gaze_people',

			color: 'hsl(227, 18%, 30%)'
		},
		{
			key: 'non_gaze_people',
			color: 'hsl(220, 33%, 96%)'
		}
	]}
	seriesLayout="stack"
	padding={{ bottom: 30, left: 30 }}
	props={{
		xAxis: { format: (_) => '', label: 'Time' },
		yAxis: { format: 'metric', label: 'Number of People' },
		tooltip: {
			header: { format: 'none' }
		},
		bars: {
			stroke: 'none'
		}
	}}
/>
