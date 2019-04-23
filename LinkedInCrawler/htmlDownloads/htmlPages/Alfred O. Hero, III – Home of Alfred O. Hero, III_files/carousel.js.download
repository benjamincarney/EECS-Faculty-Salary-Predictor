var $ = jQuery.noConflict();
$(document).ready(function(){    
                                   
        // Carousel Tweaks

        $('.ngg-galleryoverview').wrapInner('<div class="ngg-inner" />');
        $('.ngg-inner').wrapInner('<div class="ngg-thumbnails" />');
        $('.ngg-inner').before('<div class="prev">Prev</div>');
        $('.ngg-inner').after('<div class="next">Next</div>');
        
        $('.ngg-galleryoverview .prev').hide();
        
        $('.ngg-galleryoverview .prev').click(function() {
                var parent = $(this).parent();
                var mover = $(parent).find('.ngg-thumbnails');
                $(mover).animate({marginLeft: "+=150px"});
                $(this).fadeOut();
                $('.ngg-galleryoverview .next').fadeIn();
        });
        $('.ngg-galleryoverview .next').click(function() {
                var parent = $(this).parent();
                var mover = $(parent).find('.ngg-thumbnails');
                $(mover).animate({marginLeft: "-=150px"});
                $(this).fadeOut();
                $('.ngg-galleryoverview .prev').fadeIn();
        });
        
        
        // Theater Tweaks

        $('.nivo_slider > .nivo-caption').each(function() {
                $(this).parent().height('+=' + $(this).height());
                $(this).css('position', 'absolute');
        });

         // open external links in new window
        $('a[href^="http"]').each(function(){
                if( (this.href.indexOf(location.hostname) == -1) && (this.href.indexOf('engin.umich.edu') == -1) ) {
                        $(this).addClass( 'external' ).attr( 'target', '_blank' );
                }
        });

});
