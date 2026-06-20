<?php
/**
 * Plugin Name:       Fence Stain Simulator
 * Plugin URI:        https://ninjafencestaining.com/fence-stain-simulator
 * Description:       AI-powered fence detection and color staining preview. Drop a photo, detect the fence boards, and preview any stain color in seconds. Embed anywhere with [fence_simulator]. Rendered inside Shadow DOM so theme styles cannot override.
 * Version:           2.0.0
 * Requires at least: 5.5
 * Requires PHP:      7.4
 * Author:            TechnoTaau
 * License:           Proprietary
 * Text Domain:       fence-stain-simulator
 */

if (!defined('ABSPATH')) {
    exit;
}

define('FSV_PLUGIN_VERSION', '2.0.0');
define('FSV_PLUGIN_DIR', plugin_dir_path(__FILE__));
define('FSV_PLUGIN_URL', plugin_dir_url(__FILE__));

/**
 * Does the current singular view contain the [fence_simulator] shortcode?
 * Checks BOTH classic post_content AND Elementor's _elementor_data postmeta —
 * Elementor stores page contents as JSON in postmeta, so the standard
 * has_shortcode() call against post_content always returns false on
 * Elementor-built pages.
 */
function fsv_page_uses_shortcode() {
    if (!is_singular()) {
        return false;
    }
    $post = get_post();
    if (!$post) {
        return false;
    }
    if (has_shortcode($post->post_content, 'fence_simulator')) {
        return true;
    }
    $elementor_data = get_post_meta($post->ID, '_elementor_data', true);
    if (is_string($elementor_data) && strpos($elementor_data, 'fence_simulator') !== false) {
        return true;
    }
    return false;
}

/**
 * Register assets EARLY (priority 5) so the shortcode handler can safely
 * enqueue them later even when the shortcode is rendered from a builder
 * widget after the head has been output.
 *
 * Shadow DOM architecture:
 *   - app.css + fence-simulator-body.html are FETCHED by fsv-loader.js
 *     and injected inside the custom element's shadow root. We do NOT
 *     register app.css as a wp_register_style — it must never appear in
 *     the global document, since that's exactly what we're isolating
 *     against.
 *   - Google Fonts and Bootstrap Icons ARE enqueued at the document level.
 *     Modern browsers propagate document-level @font-face declarations
 *     into Shadow DOM, so this is the cleanest way to make fonts available
 *     inside the simulator.
 *   - app.js (defines window.FSV_initFenceSimulator) loads first.
 *   - fsv-loader.js (defines <fence-simulator> custom element) loads after
 *     app.js, with the asset URLs injected via wp_localize_script.
 */
function fsv_register_assets() {
    if (is_admin()) {
        return;
    }
    wp_register_style(
        'fsv-fonts',
        'https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=Plus+Jakarta+Sans:wght@500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap',
        array(),
        null
    );
    wp_register_style(
        'fsv-icons',
        'https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.1/font/bootstrap-icons.css',
        array(),
        '1.11.1'
    );
    wp_register_script(
        'fsv-app',
        FSV_PLUGIN_URL . 'app.js',
        array(),
        FSV_PLUGIN_VERSION,
        true
    );
    wp_register_script(
        'fsv-loader',
        FSV_PLUGIN_URL . 'fsv-loader.js',
        array('fsv-app'),
        FSV_PLUGIN_VERSION,
        true
    );
    wp_localize_script('fsv-loader', 'FSV_ASSETS', array(
        'css'   => FSV_PLUGIN_URL . 'app.css?v=' . FSV_PLUGIN_VERSION,
        'html'  => FSV_PLUGIN_URL . 'fence-simulator-body.html?v=' . FSV_PLUGIN_VERSION,
        // Injected INSIDE the shadow root by fsv-loader.js. Bootstrap Icons
        // MUST live inside the shadow because its CSS contains class rules
        // like `.bi-X::before { content: "..." }` that document-level CSS
        // cannot match against shadow-root elements.
        'fonts' => 'https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=Plus+Jakarta+Sans:wght@500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap',
        'icons' => 'https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.1/font/bootstrap-icons.css',
    ));
}
add_action('wp_enqueue_scripts', 'fsv_register_assets', 5);

/**
 * Enqueue assets when the page actually uses the shortcode.
 */
function fsv_enqueue_assets() {
    if (is_admin()) {
        return;
    }
    if (!fsv_page_uses_shortcode()) {
        return;
    }
    wp_enqueue_style('fsv-fonts');
    wp_enqueue_style('fsv-icons');
    wp_enqueue_script('fsv-app');
    wp_enqueue_script('fsv-loader');
}
add_action('wp_enqueue_scripts', 'fsv_enqueue_assets', 999);

/**
 * Preconnect to Google Fonts to warm DNS/TLS before the CSS link parses.
 */
function fsv_resource_hints($urls, $relation) {
    if (!fsv_page_uses_shortcode()) {
        return $urls;
    }
    if ('preconnect' === $relation) {
        $urls[] = array('href' => 'https://fonts.googleapis.com');
        $urls[] = array(
            'href'        => 'https://fonts.gstatic.com',
            'crossorigin' => 'anonymous',
        );
    }
    return $urls;
}
add_filter('wp_resource_hints', 'fsv_resource_hints', 10, 2);

/**
 * Shortcode handler — outputs the <fence-simulator> custom element tag.
 *
 * Also enqueues the assets as a safety net (page builders like Elementor /
 * Divi / Beaver Builder store content outside post_content and don't
 * trigger fsv_enqueue_assets() reliably). Late-stage wp_enqueue_*() still
 * works because styles are emitted via wp_footer() and scripts are
 * registered with in_footer=true.
 */
function fsv_shortcode_handler($atts = array(), $content = null) {
    wp_enqueue_style('fsv-fonts');
    wp_enqueue_style('fsv-icons');
    wp_enqueue_script('fsv-app');
    wp_enqueue_script('fsv-loader');

    return '<fence-simulator></fence-simulator>';
}
add_shortcode('fence_simulator', 'fsv_shortcode_handler');
