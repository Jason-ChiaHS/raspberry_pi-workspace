<script lang="ts">
	import * as Card from '$lib/components/ui/card/index';
	import { PieChart } from 'layerchart';
	interface GazeLivePercentageProps {
		latest_gaze_people_data: { dt_time: string; gaze_people: number; non_gaze_people: number };
	}
	let { latest_gaze_people_data }: GazeLivePercentageProps = $props();
	let percentage_of_gaze_people = $derived.by(() => {
		let total_people =
			latest_gaze_people_data.gaze_people + latest_gaze_people_data.non_gaze_people;
		if (total_people == 0) {
			return 0;
		}
		return (latest_gaze_people_data.gaze_people / total_people) * 100;
	});
</script>

<Card.Root class="h-full flex items-center">
	<div class="grid grid-cols-2 gap-1 p-2">
		<div class="h-[100px] w-[100px]">
			<PieChart
				data={[{ key: 'Average percentage of Gaze People', value: percentage_of_gaze_people }]}
				key="key"
				value="value"
				maxValue={100}
				outerRadius={-25}
				innerRadius={-20}
				cornerRadius={10}
				cRange={['hsl(215, 16%, 47%)']}
			/>
		</div>
		<div>
			<div class="text-2xl font-bold text-slate-500">
				{Math.trunc(percentage_of_gaze_people * 100) / 100}%
			</div>
			<div class="text-wrap text-lg">of people are looking</div>
		</div>
	</div>
</Card.Root>
