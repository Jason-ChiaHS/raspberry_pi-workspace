<script lang="ts">
	import DemoConfigEditor from '$lib/pages/DemoConfigEditor/index.svelte';
	import { demos } from '$lib/demos';
</script>

<DemoConfigEditor demo={demos['helmet']} />
