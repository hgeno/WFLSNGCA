% resizes the image and saves it to fname
function saveimg(hfig, fname)

	% set the units to use
	set(hfig, 'PaperUnits', 'centimeters');
	
	% get the size of the current paper
	papersize = get(hfig, 'PaperSize');
	
	% new size
	width = 21;
	height = floor(width*2/3);
	
	% calculate the position to center the images
	left = papersize(1)/2 - width/2;
	bottom = papersize(2)/2 - height/2;
	
	% set the figure size and print
	figuresize = [left bottom, width, height];
	set(hfig, 'PaperPosition', figuresize);
	
    % cd 
%     cd(savefolder);
	% save
	saveas(hfig, sprintf('%s', fname), 'epsc');
	saveas(hfig, sprintf('%s.fig', fname));
%     saveas(hfig, sprintf('%s.csv', fname),'csv');
	
	print(hfig, sprintf('%s', fname), '-dpng', '-r600');
	
    % cd
    
%     cd ..
	%keyboard

	%saveas(hfig, sprintf('%s.emf', fname), 'emf');
end
