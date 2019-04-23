var $ = jQuery.noConflict();
$(document).ready(function(){    
    $("#content table > tbody > tr:odd").addClass("odd");
    $("#content table > tbody > tr:not(.odd)").addClass("even"); 

    $("#home-links .home-link:odd").addClass("odd");
    $("#home-links .home-link:not(.odd)").addClass("even");

    // tablesorter:
    $("#content table").tablesorter({
        sortReset: true, // reset column to original unsorted state on third click
        widgets: ["zebra"] // maintain color striping when the order changes via sort
    });
});    
