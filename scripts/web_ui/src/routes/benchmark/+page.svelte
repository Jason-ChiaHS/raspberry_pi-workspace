<script lang="ts">
	import io from 'socket.io-client';
	import { onMount } from 'svelte';
	import { APP_VERSION } from '$lib/vars';
	import * as Resizable from '$lib/components/ui/resizable/index.js';
	import {Button} from "$lib/components/ui/button/index";

	let tt: {server_time: string, client_time: string, size: number}[] = $state([]);
	let frame_times: number[] = $state([]);
	let rtt = $state("");
	let fps = $derived(
		frame_times.length > 15
			? `FPS: ${(frame_times.length / (frame_times[frame_times.length - 1] - frame_times[0])) * 1000}`
			: 'Waiting for Demo to start...'
	);

	function push_frame_time(frame_time: number) {
		frame_times.push(frame_time);
		if (frame_times.length > 30) {
			frame_times.shift();
		}
	}

	onMount(() => {
		const socket = io(`http://${window.location.hostname}:9000/image_meta`);
		socket.on('recv_image_meta', (data) => {
			// {json of image and metadata} data
			let datetime_str = data.time;
			let datetime_iso_str = datetime_str.slice(0, 23).replace(" ", "T"); // truncate to milliseconds
			let dt = new Date(datetime_iso_str);
			let now = Date.now();
			let diff = now - dt.getTime();
			rtt = `RTT: ${diff/1000}`
			let size = new Blob([data]).size;
			tt.push({server_time: dt.toISOString(), client_time: new Date(now).toISOString(), size: size});

			let current_frame_time = performance.now();
			push_frame_time(current_frame_time);
		});
	});
	function to_csv(){
		let data = [];
		data.push(["server_time", "client_time", "size"]);
		tt.map(row => {data.push([row.server_time, row.client_time, row.size])});
		return data.map(row => row.join(",")).join("\n");
	}
	function download_csv() {
		const csv = to_csv();
		const blob = new Blob([csv], { type: "text/csv" });
		const url = URL.createObjectURL(blob);

		const a = document.createElement("a");
		a.href = url;
		a.download = "data.csv";
		document.body.appendChild(a);
		a.click();
		document.body.removeChild(a);
		URL.revokeObjectURL(url);
	}

</script>

<div class="h-dvh">
	<div class="grid h-full w-full grid-cols-5 grid-rows-5 gap-1 md:p-2">
		<div class="col-span-1 row-span-5 flex flex-col gap-1 rounded-md border-2 p-1">
			<div class="border-2">
				<p>{fps}</p>
				<p>{rtt}</p>
				<Button onclick={download_csv} >Save CSV</Button>
			</div>
		</div>
		<div class="relative col-span-4 row-span-5 rounded-md border-2 p-2">
            <p> Wow </p>
		</div>
	</div>
</div>
