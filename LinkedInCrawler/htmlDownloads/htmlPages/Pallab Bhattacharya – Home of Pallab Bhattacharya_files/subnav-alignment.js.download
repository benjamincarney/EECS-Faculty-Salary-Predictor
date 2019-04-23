/**
 * Author: Rachel Leggett
 * Adds class for when subnav is hidden so #primary can have width: 100% only in that case
 **/

var $ = jQuery.noConflict();
$(document).ready(function() {
    if( !( $('#subnav .menu-item.current-menu-item').hasClass('current-menu-ancestor') ) &&
        !( $('#subnav .menu-item.current-menu-item').hasClass('menu-item-has-children') ) &&
        !( $('#subnav .menu-item.current-menu-ancestor').hasClass('menu-item-has-children') ) ) {
        $('#secondary').addClass('no-subnav');
        $('#primary').addClass('no-subnav');
    }
});
