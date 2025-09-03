<script lang="ts">
	import LoaderCircle from '@lucide/svelte/icons/loader-circle';
	import { Button } from '$lib/components/ui/button/index';
	import { toast } from 'svelte-sonner';

	let loading = $state(false);
</script>

<Button
	variant="link"
	disabled={loading}
	onclick={() => {
		loading = true;
		fetch('/api/start_raspi_cam_srv', { method: 'POST' }).then((res) => {
		    loading = false;
            if (res.ok){
		    window.open(`http://${window.location.hostname}:9001`, '_blank');
            } else{
				toast.error('Error starting camera config');
            }
		});
	}}
	class="text-headerText font-bold"
	>Camera Config
	{#if loading}
		<LoaderCircle class="animate-spin" />
	{:else}
		<LoaderCircle class="hidden animate-spin" />
	{/if}
</Button>
